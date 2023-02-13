/*
 * TensorFlow interface helpers.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"

namespace tensorflow {

  void setLogging(const std::string& level) { setenv("TF_CPP_MIN_LOG_LEVEL", level.c_str(), 0); }

  void setThreading(SessionOptions& sessionOptions, int nThreads) {
    // set number of threads used for intra and inter operation communication
    sessionOptions.config.set_intra_op_parallelism_threads(nThreads);
    sessionOptions.config.set_inter_op_parallelism_threads(nThreads);
  }

  void setThreading(SessionOptions& sessionOptions, int nThreads, const std::string& singleThreadPool) {
    edm::LogInfo("PhysicsTools/TensorFlow") << "setting the thread pool via tensorflow::setThreading() is deprecated";

    setThreading(sessionOptions, nThreads);
  }

  void setBackend(SessionOptions& sessionOptions, Backend backend) {
    /*
     * The TensorFlow backend configures the available devices using options provided in the sessionOptions proto.
     * // Options from https://github.com/tensorflow/tensorflow/blob/c53dab9fbc9de4ea8b1df59041a5ffd3987328c3/tensorflow/core/protobuf/config.proto
     *
     * If the device_count["GPU"] = 0 GPUs are not used. 
     * The visible_device_list configuration is used to map the `visible` devices (from CUDA_VISIBLE_DEVICES) to `virtual` devices.
     * If Backend::cpu is request, the GPU device is disallowed by device_count configuration.
     * If Backend::cuda is request:
     *  - if ResourceInformation shows an available Nvidia GPU device:
     *     the device is used with memory_growth configuration (not allocating all cuda memory at once).
     *  - if no device is present: an exception is raised.
     */

    edm::Service<edm::ResourceInformation> ri;
    if (backend == Backend::cpu) {
      // disable GPU usage
      (*sessionOptions.config.mutable_device_count())["GPU"] = 0;
      sessionOptions.config.mutable_gpu_options()->set_visible_device_list("");
    }
    // NVidia GPU
    else if (backend == Backend::cuda) {
      if (not ri->nvidiaDriverVersion().empty()) {
        // Take only the first GPU in the CUDA_VISIBLE_DEVICE list
        (*sessionOptions.config.mutable_device_count())["GPU"] = 1;
        sessionOptions.config.mutable_gpu_options()->set_visible_device_list("0");
        // Do not allocate all the memory on the GPU at the beginning.
        sessionOptions.config.mutable_gpu_options()->set_allow_growth(true);
      } else {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "Cuda backend requested, but no NVIDIA GPU available in the job";
        ex.addContext("Calling tensorflow::setBackend()");
        throw ex;
      }
    }
    // ROCm and Intel GPU are still not supported
    else if ((backend == Backend::rocm) || (backend == Backend::intel)) {
      edm::Exception ex(edm::errors::UnavailableAccelerator);
      ex << "ROCm/Intel GPU backend requested, but TF is not compiled yet for this platform";
      ex.addContext("Calling tensorflow::setBackend()");
      throw ex;
    }
    // Get NVidia GPU if possible or fallback to CPU
    else if (backend == Backend::best) {
      // Check if a Nvidia GPU is availabl
      if (not ri->nvidiaDriverVersion().empty()) {
        // Take only the first GPU in the CUDA_VISIBLE_DEVICE list
        (*sessionOptions.config.mutable_device_count())["GPU"] = 1;
        sessionOptions.config.mutable_gpu_options()->set_visible_device_list("0");
        // Do not allocate all the memory on the GPU at the beginning.
        sessionOptions.config.mutable_gpu_options()->set_allow_growth(true);
      } else {
        // Just CPU support
        (*sessionOptions.config.mutable_device_count())["GPU"] = 0;
        sessionOptions.config.mutable_gpu_options()->set_visible_device_list("");
      }
    }
  }

  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir, const std::string& tag, SessionOptions& sessionOptions) {
    // objects to load the graph
    Status status;
    RunOptions runOptions;
    SavedModelBundle bundle;

    // load the model
    status = LoadSavedModel(sessionOptions, runOptions, exportDir, {tag}, &bundle);
    if (!status.ok()) {
      throw cms::Exception("InvalidMetaGraphDef")
          << "error while loading metaGraphDef from '" << exportDir << "': " << status.ToString();
    }

    // return a copy of the graph
    return new MetaGraphDef(bundle.meta_graph_def);
  }

  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, SessionOptions& sessionOptions) {
    edm::LogInfo("PhysicsTools/TensorFlow")
        << "tensorflow::loadMetaGraph() is deprecated, use tensorflow::loadMetaGraphDef() instead";

    return loadMetaGraphDef(exportDir, tag, sessionOptions);
  }

  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir, const std::string& tag, Backend backend, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);
    setBackend(sessionOptions, backend);

    return loadMetaGraphDef(exportDir, tag, sessionOptions);
  }

  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, Backend backend, int nThreads) {
    edm::LogInfo("PhysicsTools/TensorFlow")
        << "tensorflow::loadMetaGraph() is deprecated, use tensorflow::loadMetaGraphDef() instead";

    return loadMetaGraphDef(exportDir, tag, backend, nThreads);
  }

  GraphDef* loadGraphDef(const std::string& pbFile) {
    // objects to load the graph
    Status status;

    // load it
    GraphDef* graphDef = new GraphDef();
    status = ReadBinaryProto(Env::Default(), pbFile, graphDef);

    // check for success
    if (!status.ok()) {
      throw cms::Exception("InvalidGraphDef")
          << "error while loading graphDef from '" << pbFile << "': " << status.ToString();
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

  Session* createSession(Backend backend, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);
    setBackend(sessionOptions, backend);

    return createSession(sessionOptions);
  }

  Session* createSession(const MetaGraphDef* metaGraphDef,
                         const std::string& exportDir,
                         SessionOptions& sessionOptions) {
    // check for valid pointer
    if (metaGraphDef == nullptr) {
      throw cms::Exception("InvalidMetaGraphDef") << "error while creating session: metaGraphDef is nullptr";
    }

    // check that the graph has nodes
    if (metaGraphDef->graph_def().node_size() <= 0) {
      throw cms::Exception("InvalidMetaGraphDef") << "error while creating session: graphDef has no nodes";
    }

    Session* session = createSession(sessionOptions);

    // add the graph def from the meta graph
    Status status;
    status = session->Create(metaGraphDef->graph_def());
    if (!status.ok()) {
      throw cms::Exception("InvalidMetaGraphDef")
          << "error while attaching metaGraphDef to session: " << status.ToString();
    }

    // restore variables using the variable and index files in the export directory
    // first, find names and paths
    std::string varFileTensorName = metaGraphDef->saver_def().filename_tensor_name();
    std::string restoreOpName = metaGraphDef->saver_def().restore_op_name();
    std::string varDir = io::JoinPath(exportDir, kSavedModelVariablesDirectory);
    std::string indexFile = io::JoinPath(varDir, MetaFilename(kSavedModelVariablesFilename));
    std::string varFile = io::JoinPath(varDir, kSavedModelVariablesFilename);

    // when the index file is missing, there's nothing to do
    if (!Env::Default()->FileExists(indexFile).ok()) {
      return session;
    }

    // create a tensor to store the variable file
    Tensor varFileTensor(DT_STRING, TensorShape({}));
    varFileTensor.scalar<tensorflow::tstring>()() = varFile;

    // run the restore op
    status = session->Run({{varFileTensorName, varFileTensor}}, {}, {restoreOpName}, nullptr);
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while restoring variables in session: " << status.ToString();
    }

    return session;
  }

  Session* createSession(const MetaGraphDef* metaGraphDef, const std::string& exportDir, Backend backend, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);
    setBackend(sessionOptions, backend);

    return createSession(metaGraphDef, exportDir, sessionOptions);
  }

  Session* createSession(const GraphDef* graphDef, SessionOptions& sessionOptions) {
    // check for valid pointer
    if (graphDef == nullptr) {
      throw cms::Exception("InvalidGraphDef") << "error while creating session: graphDef is nullptr";
    }

    // check that the graph has nodes
    if (graphDef->node_size() <= 0) {
      throw cms::Exception("InvalidGraphDef") << "error while creating session: graphDef has no nodes";
    }

    // create a new, empty session
    Session* session = createSession(sessionOptions);

    // add the graph def
    Status status;
    status = session->Create(*graphDef);

    // check for success
    if (!status.ok()) {
      throw cms::Exception("InvalidSession") << "error while attaching graphDef to session: " << status.ToString();
    }

    return session;
  }

  Session* createSession(const GraphDef* graphDef, Backend backend, int nThreads) {
    // create session options and set thread options
    SessionOptions sessionOptions;
    setThreading(sessionOptions, nThreads);
    setBackend(sessionOptions, backend);

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

  bool closeSession(const Session*& session) {
    auto s = const_cast<Session*>(session);
    bool state = closeSession(s);

    // reset the pointer
    session = nullptr;

    return state;
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

  void SessionCache::closeSession() {
    // delete the session if set
    Session* s = session.load();
    if (s != nullptr) {
      tensorflow::closeSession(s);
      session.store(nullptr);
    }

    // delete the graph if set
    if (graph.load() != nullptr) {
      delete graph.load();
      graph.store(nullptr);
    }
  }

}  // namespace tensorflow
