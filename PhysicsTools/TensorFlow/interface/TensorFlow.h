/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 2.1.
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

  typedef std::pair<std::string, Tensor> NamedTensor;
  typedef std::vector<NamedTensor> NamedTensorList;

  // set the tensorflow log level
  void setLogging(const std::string& level = "3");

  // updates the config of sessionOptions so that it uses nThreads
  void setThreading(SessionOptions& sessionOptions, int nThreads = 1);

  // deprecated
  // updates the config of sessionOptions so that it uses nThreads, prints a deprecation warning
  // since the threading configuration is done per run() call as of 2.1
  void setThreading(SessionOptions& sessionOptions, int nThreads, const std::string& singleThreadPool);

  // loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
  // predefined sessionOptions
  // transfers ownership
  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir, const std::string& tag, SessionOptions& sessionOptions);

  // deprecated in favor of loadMetaGraphDef
  MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag, SessionOptions& sessionOptions);

  // loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
  // nThreads
  // transfers ownership
  MetaGraphDef* loadMetaGraphDef(const std::string& exportDir,
                                 const std::string& tag = kSavedModelTagServe,
                                 int nThreads = 1);

  // deprecated in favor of loadMetaGraphDef
  MetaGraphDef* loadMetaGraph(const std::string& exportDir,
                              const std::string& tag = kSavedModelTagServe,
                              int nThreads = 1);

  // loads a graph definition saved as a protobuf file at pbFile
  // transfers ownership
  GraphDef* loadGraphDef(const std::string& pbFile);

  // return a new, empty session using predefined sessionOptions
  // transfers ownership
  Session* createSession(SessionOptions& sessionOptions);

  // return a new, empty session with nThreads
  // transfers ownership
  Session* createSession(int nThreads = 1);

  // return a new session that will contain an already loaded meta graph whose exportDir must be
  // given in order to load and initialize the variables, sessionOptions are predefined
  // an error is thrown when metaGraphDef is a nullptr or when the graph has no nodes
  // transfers ownership
  Session* createSession(MetaGraphDef* metaGraphDef, const std::string& exportDir, SessionOptions& sessionOptions);

  // return a new session that will contain an already loaded meta graph whose exportDir must be given
  // in order to load and initialize the variables, threading options are inferred from nThreads
  // an error is thrown when metaGraphDef is a nullptr or when the graph has no nodes
  // transfers ownership
  Session* createSession(MetaGraphDef* metaGraphDef, const std::string& exportDir, int nThreads = 1);

  // return a new session that will contain an already loaded graph def, sessionOptions are predefined
  // an error is thrown when graphDef is a nullptr or when the grah has no nodes
  // transfers ownership
  Session* createSession(GraphDef* graphDef, SessionOptions& sessionOptions);

  // return a new session that will contain an already loaded graph def, threading options are
  // inferred from nThreads
  // an error is thrown when graphDef is a nullptr or when the grah has no nodes
  // transfers ownership
  Session* createSession(GraphDef* graphDef, int nThreads = 1);

  // closes a session, calls its destructor, resets the pointer, and returns true on success
  bool closeSession(Session*& session);

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool using threadPoolOptions
  // used for thread scheduling with custom thread pool options
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const thread::ThreadPoolOptions& threadPoolOptions);

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           thread::ThreadPoolInterface* threadPool);

  // run the session with inputs and outputNames, store output tensors, and control the underlying
  // thread pool using a threadPoolName ("no_threads", "tbb", or "tensorflow")
  // throws a cms exception when not successful
  void run(Session* session,
           const NamedTensorList& inputs,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName = "no_threads");

  // run the session without inputs but only outputNames, store output tensors, and control the
  // underlying thread pool using a threadPoolName ("no_threads", "tbb", or "tensorflow")
  // throws a cms exception when not successful
  void run(Session* session,
           const std::vector<std::string>& outputNames,
           std::vector<Tensor>* outputs,
           const std::string& threadPoolName = "no_threads");

}  // namespace tensorflow

#endif  // PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
