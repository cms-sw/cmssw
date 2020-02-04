/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 2.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace tensorflow {

  void setLogging(const std::string& level) { setenv("TF_CPP_MIN_LOG_LEVEL", level.c_str(), 0); }

  void setThreading(SessionOptions& sessionOptions, int nThreads) {
    // set number of threads used for intra and inter operation communication
    sessionOptions.config.set_intra_op_parallelism_threads(nThreads);
    sessionOptions.config.set_inter_op_parallelism_threads(nThreads);
  }

  void setThreading(SessionOptions& sessionOptions, int nThreads, const std::string& singleThreadPool) {
    edm::LogInfo("PhysicsTools/TensorFlow") << "setting the thread pool via tensorflow::setThreading is deprecated";

    setThreading(sessionOptions, nThreads);
  }

  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, SessionOptions& sessionOptions) {
    // objects to load the graph
    Status status;
    RunOptions runOptions;
    SavedModelBundle bundle;

    // load the model
    status = LoadSavedModel(sessionOptions, runOptions, exportDir, {tag}, &bundle);
    if (!status.ok()) {
      throw cms::Exception("InvalidMetaGraph") << "error while loading meta graph: " << status.ToString();
    }

    // return a copy of the graph
    return new MetaGraphDef(bundle.meta_graph_def);
  }

  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return loadMetaGraph(exportDir, tag, sessionOptions);
  }

  GraphDef* loadGraphDef(const std::string& pbFile) {
    // objects to load the graph
    Status status;

    // load it
    GraphDef* graphDef = new GraphDef();
    status = ReadBinaryProto(Env::Default(), pbFile, graphDef);

    // check for success
    if (!status.ok()) {
      throw cms::Exception("InvalidGraphDef") << "error while loading graph def: " << status.ToString();
    }

    return graphDef;
  }

  Session* createSession(SessionOptions& sessionOptions) {
    // objects to create the session
    Status status;

    // create a new, empty session
    Session* session = nullptr;
    status = NewSession(sessionOptions, &session);
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while creating session: " << status.ToString();
    }

    return session;
  }

  Session* createSession(int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(sessionOptions);
  }

  Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir, SessionOptions& sessionOptions) {
    Session* session = createSession(sessionOptions);

    // add the graph def from the meta graph
    Status status;
    status = session->Create(metaGraph->graph_def());
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while attaching meta graph to session: " << status.ToString();
    }

    // restore variables using the variable and index files in the export directory
    // first, find names and paths
    std::string varFileTensorName = metaGraph->saver_def().filename_tensor_name();
    std::string restoreOpName = metaGraph->saver_def().restore_op_name();
    std::string varDir = io::JoinPath(exportDir, kSavedModelVariablesDirectory);
    std::string indexFile = io::JoinPath(varDir, MetaFilename(kSavedModelVariablesFilename));
    std::string varFile = io::JoinPath(varDir, kSavedModelVariablesFilename);

    // when the index file is missing, there's nothing to do
    if (!Env::Default()->FileExists(indexFile).ok()) {
      return session;
    }

    // create a tensor to store the variable file
    Tensor varFileTensor(DT_STRING, TensorShape({}));
    varFileTensor.scalar<std::string>()() = varFile;

    // run the restore op
    status = session->Run({{varFileTensorName, varFileTensor}}, {}, {restoreOpName}, nullptr);
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while restoring variables in session: " << status.ToString();
    }

    return session;
  }

  Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(metaGraph, exportDir, sessionOptions);
  }

  Session* createSession(GraphDef* graphDef, SessionOptions& sessionOptions) {
    // create a new, empty session
    Session* session = createSession(sessionOptions);

    // add the graph def
    Status status;
    status = session->Create(*graphDef);

    // check for success
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while attaching graph def to session: " << status.ToString();
    }

    return session;
  }

  Session* createSession(GraphDef* graphDef, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);

    return createSession(graphDef, sessionOptions);
  }

  bool closeSession(Session*& session) {
    if (session == nullptr) {
      return true;
    }

    // close and delete the session
    Status status = session->Close();
    delete session;

    // reset the pointer
    session = nullptr;

    return status.ok();
  }

  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const thread::ThreadPoolOptions& threadPoolOptions) {
    if (session == nullptr) {
      throw cms::Exception("InvalidSession") << "cannot run empty session";
    }

    // create empty run options
    RunOptions runOptions;

    // run and check the status
    Status status = session->Run(runOptions, inputs, outputNames, {}, outputs, nullptr, threadPoolOptions);
    if (!status.ok()) {
      throw cms::Exception("InvalidRun") << "error while running session: " << status.ToString();
    }
  }

  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           thread::ThreadPoolInterface* threadPool) {
    // create thread pool options
    thread::ThreadPoolOptions threadPoolOptions;
    threadPoolOptions.inter_op_threadpool = threadPool;
    threadPoolOptions.intra_op_threadpool = threadPool;

    // run
    run(session, inputs, outputNames, outputs, threadPoolOptions);
  }

  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName) {
    // lookup the thread pool and forward the call accordingly
    if (threadPoolName == "no_threads") {
      run(session, inputs, outputNames, outputs, &NoThreadPool::instance());
    } else if (threadPoolName == "tbb") {
      // the TBBTreadPool singleton should be already initialized before with a number of threads
      run(session, inputs, outputNames, outputs, &TBBThreadPool::instance());
    } else if (threadPoolName == "tensorflow") {
      run(session, inputs, outputNames, outputs, nullptr);
    } else {
      throw cms::Exception("UnknownThreadPool")
          << "thread pool implementation'" << threadPoolName << "' unknown, use 'no_threads', 'tbb', or 'tensorflow'";
    }
  }

  void run(Session* session,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName) {
    run(session, {}, outputNames, outputs, threadPoolName);
  }

}  // namespace tensorflow
