import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

st.title("RAAFYA AI")

def vector_embedding():
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-small")
        st.session_state.docs = []
        for uploaded_file in uploaded_files:
            # Load each PDF file
            loader = PyPDFLoader(uploaded_file)
            docs = loader.load()
            st.session_state.docs.extend(docs)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=500
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs)
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )
        st.write("Vectorstore created successfully!")


with st.sidebar:
    st.title("Configurations")

    st.subheader("Model Selection")
    model_options = [
        "Gemma-7b-It",
        "Llama3-70b-8192",
        "Llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4-turbo"
    ]
    selected_model = st.selectbox("Choose a model:", model_options)

    st.subheader("System Prompt (Optional)")
    system_template = st.text_area("Provide instructions to the model", placeholder="Example: You are a helpful and informative AI assistant.")
    
    # Apply button for system prompt
    apply_system_prompt = st.button("Apply", disabled=not system_template)

    st.subheader("Temperature")
    temperature = st.slider(
        "Temperature (Higher number means more creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )

    # Checkbox in the sidebar
    talk_to_docs = st.checkbox("Talk to your documents")

    # Only show file uploader if the checkbox is ticked
    if talk_to_docs:
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True
        )

        # Add to Knowledge Base button
        if uploaded_files:
            if st.button("Add to knowledge base"):
                vector_embedding()
        else:
            st.write("Please upload PDFs first.")

# Create LLM instance based on selected model
if selected_model == "gpt-3.5-turbo":
    llm = ChatOpenAI(model_name=selected_model, temperature=temperature)
else:
    llm = ChatGroq(temperature=temperature, model_name=selected_model)


prompt = ChatPromptTemplate.from_template(
    """

    {system_template}

    {context}

    Question: {input}
    """
)

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
    st.session_state.embeddings = None
    st.session_state.docs = []


# Chat input is always showing
user_query = st.chat_input("Talk to your documents", disabled=talk_to_docs and st.session_state.vectors is None)

if user_query and st.session_state.vectors is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": user_query, "system_template": system_template if apply_system_prompt else ""})
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------")

#  Disable the chat input if the checkbox is ticked but no files are uploaded
elif talk_to_docs and st.session_state.vectors is None:
    st.write("Please upload files and add them to the knowledge base first.")