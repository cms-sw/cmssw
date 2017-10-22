/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 1.3.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

namespace tensorflow
{

void setLogging(const std::string& level)
{
    setenv("TF_CPP_MIN_LOG_LEVEL", level.c_str(), 0);
}

void setThreading(SessionOptions& sessionOptions, int nThreads)
{
    // set number of threads used for intra and inter operation communication
    sessionOptions.config.set_intra_op_parallelism_threads(nThreads);
    sessionOptions.config.set_inter_op_parallelism_threads(nThreads);

    // when exactly one thread is requested use the custom session factory
    if (nThreads == 1)
    {
        sessionOptions.target = "no_threads";
    }
}

MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, int nThreads)
{
    // objects to load the graph
    Status status;
    SessionOptions sessionOptions;
    RunOptions runOptions;
    SavedModelBundle bundle;

    // set thread options
    setThreading(sessionOptions, nThreads);

    // load the model
    status = LoadSavedModel(sessionOptions, runOptions, exportDir, { tag }, &bundle);
    if (!status.ok())
    {
        throw cms::Exception("InvalidGraph")
            << "error while loading graph: " << status.ToString();
    }

    // return a copy
    return new MetaGraphDef(bundle.meta_graph_def);
}

Session* createSession(int nThreads)
{
    // objects to create the session
    Status status;
    SessionOptions sessionOptions;

    // set thread options
    setThreading(sessionOptions, nThreads);

    // create a new, empty session
    Session* session = nullptr;
    status = NewSession(sessionOptions, &session);
    if (!status.ok())
    {
        throw cms::Exception("InvalidSession")
            << "error while creating session: " << status.ToString();
    }

    return session;
}

Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir, int nThreads)
{
    Session* session = createSession(nThreads);

    // add the graph def from the meta graph
    Status status;
    status = session->Create(metaGraph->graph_def());
    if (!status.ok())
    {
        throw cms::Exception("InvalidSession")
            << "error while attaching graph to session: " << status.ToString();
    }

    // restore variables using the variable and index files in the export directory
    // first, find names and paths
    std::string varFileTensorName = metaGraph->saver_def().filename_tensor_name();
    std::string restoreOpName = metaGraph->saver_def().restore_op_name();
    std::string varDir = io::JoinPath(exportDir, kSavedModelVariablesDirectory);
    std::string indexFile = io::JoinPath(varDir, MetaFilename(kSavedModelVariablesFilename));
    std::string varFile = io::JoinPath(varDir, kSavedModelVariablesFilename);

    // when the index file is missing, there's nothing to do
    if (!Env::Default()->FileExists(indexFile).ok())
    {
        return session;
    }

    // create a tensor to store the variable file
    Tensor varFileTensor(DT_STRING, TensorShape({}));
    varFileTensor.scalar<std::string>()() = varFile;

    // run the restore op
    status = session->Run({ { varFileTensorName, varFileTensor } }, {}, { restoreOpName }, nullptr);
    if (!status.ok())
    {
        throw cms::Exception("InvalidSession")
            << "error while restoring variables in session: " << status.ToString();
    }

    return session;
}

bool closeSession(Session*& session)
{
    if (session == nullptr)
    {
        return true;
    }

    // close and delete the session
    Status status = session->Close();
    delete session;

    // reset the pointer
    session = nullptr;

    return status.ok();
}

void run(Session* session, const NamedTensorList& inputs,
    const std::vector<std::string>& outputNames, const std::vector<std::string>& targetNodes,
    std::vector<Tensor>* outputs)
{
    if (session == nullptr)
    {
        throw cms::Exception("InvalidSession") << "cannot run empty session";
    }

    // run and check the status
    Status status = session->Run(inputs, outputNames, targetNodes, outputs);
    if (!status.ok())
    {
        throw cms::Exception("InvalidRun")
            << "error while running session: " << status.ToString();
    }
}

void run(Session* session, const std::vector<std::string>& inputNames,
    const std::vector<Tensor>& inputTensors, const std::vector<std::string>& outputNames,
    const std::vector<std::string>& targetNodes, std::vector<Tensor>* outputs)
{
    if (inputNames.size() != inputTensors.size())
    {
        throw cms::Exception("InvalidInput") << "numbers of input names and tensors not equal";
    }

    NamedTensorList inputs;
    for (size_t i = 0; i < inputNames.size(); i++)
    {
        inputs.push_back(NamedTensor(inputNames[i], inputTensors[i]));
    }

    run(session, inputs, outputNames, targetNodes, outputs);
}

void run(Session* session, const NamedTensorList& inputs,
    const std::vector<std::string>& outputNames, std::vector<Tensor>* outputs)
{
    run(session, inputs, outputNames, {}, outputs);
}

void run(Session* session, const std::vector<std::string>& inputNames,
    const std::vector<Tensor>& inputTensors, const std::vector<std::string>& outputNames,
    std::vector<Tensor>* outputs)
{
    run(session, inputNames, inputTensors, outputNames, {}, outputs);
}

} // namespace tensorflow
