/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 1.13.
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

void setThreading(SessionOptions& sessionOptions, int nThreads,
    const std::string& singleThreadPool)
{
    // set number of threads used for intra and inter operation communication
    sessionOptions.config.set_intra_op_parallelism_threads(nThreads);
    sessionOptions.config.set_inter_op_parallelism_threads(nThreads);

    // when exactly one thread is requested use a custom thread pool
    if (nThreads == 1 && !singleThreadPool.empty())
    {
        // check for known thread pools
        if (singleThreadPool != "no_threads" && singleThreadPool != "tbb")
        {
            throw cms::Exception("UnknownThreadPool")
                << "thread pool '" << singleThreadPool << "' unknown, use 'no_threads' or 'tbb'";
        }
        sessionOptions.target = singleThreadPool;
    }
}

MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag,
    SessionOptions& sessionOptions)
{
    // objects to load the graph
    Status status;
    RunOptions runOptions;
    SavedModelBundle bundle;

    // load the model
    status = LoadSavedModel(sessionOptions, runOptions, exportDir, { tag }, &bundle);
    if (!status.ok())
    {
        throw cms::Exception("InvalidMetaGraph")
            << "error while loading meta graph: " << status.ToString();
    }

    // return a copy of the graph
    return new MetaGraphDef(bundle.meta_graph_def);
}

MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, int nThreads)
{
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return loadMetaGraph(exportDir, tag, sessionOptions);
}

GraphDef* loadGraphDef(const std::string& pbFile)
{
    // objects to load the graph
    Status status;

    // load it
    GraphDef* graphDef = new GraphDef();
    status = ReadBinaryProto(Env::Default(), pbFile, graphDef);

    // check for success
    if (!status.ok())
    {
        throw cms::Exception("InvalidGraphDef")
            << "error while loading graph def: " << status.ToString();
    }

    return graphDef;
}

Session* createSession(SessionOptions& sessionOptions)
{
    // objects to create the session
    Status status;

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

Session* createSession(int nThreads)
{
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(sessionOptions);
}

Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir,
    SessionOptions& sessionOptions)
{
    Session* session = createSession(sessionOptions);

    // add the graph def from the meta graph
    Status status;
    status = session->Create(metaGraph->graph_def());
    if (!status.ok())
    {
        throw cms::Exception("InvalidSession")
            << "error while attaching meta graph to session: " << status.ToString();
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

Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir, int nThreads)
{
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(metaGraph, exportDir, sessionOptions);
}

Session* createSession(GraphDef* graphDef, SessionOptions& sessionOptions)
{
    // create a new, empty session
    Session* session = createSession(sessionOptions);

    // add the graph def
    Status status;
    status = session->Create(*graphDef);

    // check for success
    if (!status.ok())
    {
        throw cms::Exception("InvalidSession")
            << "error while attaching graph def to session: " << status.ToString();
    }

    return session;
}

Session* createSession(GraphDef* graphDef, int nThreads)
{
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(graphDef, sessionOptions);
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
