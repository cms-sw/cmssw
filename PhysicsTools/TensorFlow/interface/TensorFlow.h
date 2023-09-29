/*
 * TensorFlow interface helpers.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
#define PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

#include "PhysicsTools/TensorFlow/interface/NoThreadPool.h"
#include "PhysicsTools/TensorFlow/interface/TBBThreadPool.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace tensorflow {

  enum class Backend { cpu, cuda, rocm, intel, best };

  typedef std::pair<std::string, Tensor> NamedTensor;
  typedef std::vector<NamedTensor> NamedTensorList;

  struct Options {
    int _nThreads;
    Backend _backend;
    SessionOptions _options;

    Options(Backend backend) : _nThreads{1}, _backend{backend} {
      setThreading(_nThreads);
      setBackend(_backend);
    };

    Options() : _nThreads{1}, _backend{Backend::cpu} {
      setThreading(_nThreads);
      setBackend(_backend);
    };

    // updates the config of sessionOptions so that it uses nThreads
    void setThreading(int nThreads = 1);

    // Set the backend option cpu/cuda
    // The gpu memory is set to "allow_growth" to avoid TF getting all the CUDA memory at once.
    void setBackend(Backend backend = Backend::cpu);

    SessionOptions& getSessionOptions() { return _options; };
    int getNThreads() const { return _nThreads; };
    Backend getBackend() const { return _backend; };
  };

  // set the tensorflow log level
  void setLogging(const std::string& level = "3");

  // loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
  // predefined options
  // transfers ownership
  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir, const std::string& tag = kSavedModelTagServe);

  // loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
  // user provided options
  // transfers ownership
  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir, const std::string& tag, Options& options);

  // deprecated in favor of loadMetaGraphDef
  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, Options& Options);

  // loads a graph definition saved as a protobuf file at pbFile
  // transfers ownership
  GraphDef* loadGraphDef(const std::string& pbFile);

  // return a new, empty session using the predefined options
  Session* createSession();

  // return a new, empty session using user provided options
  // transfers ownership
  Session* createSession(Options& options);

  // return a new session that will contain an already loaded meta graph whose exportDir must be
  // given in order to load and initialize the variables, sessionOptions are predefined
  // an error is thrown when metaGraphDef is a nullptr or when the graph has no nodes
  // transfers ownership
  Session* createSession(const MetaGraphDef* metaGraphDef, const std::string& exportDir, Options& options);

  // return a new session that will contain an already loaded graph def, sessionOptions are predefined
  // an error is thrown when graphDef is a nullptr or when the graph has no nodes
  // transfers ownership
  Session* createSession(const GraphDef* graphDef);

  // return a new session that will contain an already loaded graph def, sessionOptions are user defined
  // an error is thrown when graphDef is a nullptr or when the graph has no nodes
  // transfers ownership
  Session* createSession(const GraphDef* graphDef, Options& options);

  // closes a session, calls its destructor, resets the pointer, and returns true on success
  bool closeSession(Session*& session);

  // version of the function above that accepts a const session
  bool closeSession(const Session*& session);

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool using threadPoolOptions
  // used for thread scheduling with custom thread pool options
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const thread::ThreadPoolOptions& threadPoolOptions);

  // version of the function above that accepts a const session
  inline void run(const Session* session,
                  const NamedTensorList& inputs,
                  const std::vector<std::string>& outputNames,
                  std::vector<Tensor>* outputs,
                  const thread::ThreadPoolOptions& threadPoolOptions) {
    // TF takes a non-const session in the run call which is, however, thread-safe and logically
    // const, thus const_cast is consistent
    run(const_cast<Session*>(session), inputs, outputNames, outputs, threadPoolOptions);
  }

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           thread::ThreadPoolInterface* threadPool);

  // version of the function above that accepts a const session
  inline void run(const Session* session,
                  const NamedTensorList& inputs,
                  const std::vector<std::string>& outputNames,
                  std::vector<Tensor>* outputs,
                  thread::ThreadPoolInterface* threadPool) {
    // TF takes a non-const session in the run call which is, however, thread-safe and logically
    // const, thus const_cast is consistent
    run(const_cast<Session*>(session), inputs, outputNames, outputs, threadPool);
  }

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool using a threadPoolName ("no_threads", "tbb", or "tensorflow")
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName = "no_threads");

  // version of the function above that accepts a const session
  inline void run(const Session* session,
                  const NamedTensorList& inputs,
                  const std::vector<std::string>& outputNames,
                  std::vector<Tensor>* outputs,
                  const std::string& threadPoolName = "no_threads") {
    // TF takes a non-const session in the run call which is, however, thread-safe and logically
    // const, thus const_cast is consistent
    run(const_cast<Session*>(session), inputs, outputNames, outputs, threadPoolName);
  }

  // run the session without inputs but only outputNames, store output tensors, and control the
  // underlying thread pool using a threadPoolName ("no_threads", "tbb", or "tensorflow")
  // throws a cms exception when not successful
  void run(Session* session,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName = "no_threads");

  // version of the function above that accepts a const session
  inline void run(const Session* session,
                  const std::vector<std::string>& outputNames,
                  std::vector<Tensor>* outputs,
                  const std::string& threadPoolName = "no_threads") {
    // TF takes a non-const session in the run call which is, however, thread-safe and logically
    // const, thus const_cast is consistent
    run(const_cast<Session*>(session), outputNames, outputs, threadPoolName);
  }

  // struct that can be used in edm::stream modules for caching a graph and a session instance,
  // both made atomic for cases where access is required from multiple threads
  struct SessionCache {
    std::atomic<GraphDef*> graph;
    std::atomic<Session*> session;

    // constructor
    SessionCache() {}

    // initializing constructor, forwarding all arguments to createSession
    template <typename... Args>
    SessionCache(const std::string& graphPath, Args&&... sessionArgs) {
      createSession(graphPath, std::forward<Args>(sessionArgs)...);
    }

    // destructor
    ~SessionCache() { closeSession(); }

    // create the internal graph representation from graphPath and the session object, forwarding
    // all additional arguments to the central tensorflow::createSession
    template <typename... Args>
    void createSession(const std::string& graphPath, Args&&... sessionArgs) {
      graph.store(loadGraphDef(graphPath));
      session.store(tensorflow::createSession(graph.load(), std::forward<Args>(sessionArgs)...));
    }

    // return a pointer to the const session
    inline const Session* getSession() const { return session.load(); }

    // closes and removes the session as well as the graph, and sets the atomic members to nullptr's
    void closeSession();
  };

}  // namespace tensorflow

#endif  // PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
