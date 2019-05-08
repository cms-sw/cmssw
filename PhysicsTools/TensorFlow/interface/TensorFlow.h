/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 1.13.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
#define PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace tensorflow
{

typedef std::pair<std::string, Tensor> NamedTensor;
typedef std::vector<NamedTensor> NamedTensorList;

// set the tensorflow log level
void setLogging(const std::string& level = "3");

// updates the config of sessionOptions so that it uses nThreads and if 1, sets the thread pool to
// singleThreadPool
void setThreading(SessionOptions& sessionOptions, int nThreads,
    const std::string& singleThreadPool = "no_threads");

// loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
// predefined sessionOptions
// transfers ownership
MetaGraphDef* loadMetaGraph(const std::string& exportDir, const std::string& tag,
    SessionOptions& sessionOptions);

// loads a meta graph definition saved at exportDir using the SavedModel interface for a tag and
// nThreads
// transfers ownership
MetaGraphDef* loadMetaGraph(const std::string& exportDir,
    const std::string& tag = kSavedModelTagServe, int nThreads = 1);

// loads a graph definition saved as a protobuf file at pbFile
// transfers ownership
GraphDef* loadGraphDef(const std::string& pbFile);

// return a new, empty session using predefined sessionOptions
// transfers ownership
Session* createSession(SessionOptions& sessionOptions);

// return a new, empty session with nThreads
// transfers ownership
Session* createSession(int nThreads = 1);

// return a new session that will contain an already loaded meta graph whose exportDir must be given
// in order to load and initialize the variables, sessionOptions are predefined
// transfers ownership
Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir,
    SessionOptions& sessionOptions);

// return a new session that will contain an already loaded meta graph whose exportDir must be given
// in order to load and initialize the variables, threading options are inferred from nThreads
// transfers ownership
Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir, int nThreads = 1);

// return a new session that will contain an already loaded graph def, sessionOptions are predefined
// transfers ownership
Session* createSession(GraphDef* graphDef, SessionOptions& sessionOptions);

// return a new session that will contain an already loaded graph def, threading options are
// inferred from nThreads
// transfers ownership
Session* createSession(GraphDef* graphDef, int nThreads = 1);

// closes a session, calls its destructor, resets the pointer, and returns true on success
bool closeSession(Session*& session);

// run the session with inputs, outputNames and targetNodes, and store output tensors
// throws a cms exception when not successful
void run(Session* session, const NamedTensorList& inputs,
    const std::vector<std::string>& outputNames, const std::vector<std::string>& targetNodes,
    std::vector<Tensor>* outputs);

// run the session with inputNames, inputTensors, outputNames and targetNodes, and store output
// tensors
// throws a cms exception when not successful
void run(Session* session, const std::vector<std::string>& inputNames,
    const std::vector<Tensor>& inputTensors, const std::vector<std::string>& outputNames,
    const std::vector<std::string>& targetNodes, std::vector<Tensor>* outputs);

// run the session with inputs and outputNames, and store output tensors
// throws a cms exception when not successful
void run(Session* session, const NamedTensorList& inputs,
    const std::vector<std::string>& outputNames, std::vector<Tensor>* outputs);

// run the session with inputNames, inputTensors and outputNames, and store output tensors
// throws a cms exception when not successful
void run(Session* session, const std::vector<std::string>& inputNames,
    const std::vector<Tensor>& inputTensors, const std::vector<std::string>& outputNames,
    std::vector<Tensor>* outputs);

} // namespace tensorflow

#endif // PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
