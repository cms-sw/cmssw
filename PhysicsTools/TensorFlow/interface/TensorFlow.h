/*
 * TensorFlow interface helpers.
 * Based on TensorFlow C++ API 1.3.
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

// updates the config of sessionOptions so that it uses a single thread
void setThreading(SessionOptions& sessionOptions, int nThreads);

// loads a meta graph definition saved at exportDir using the SavedModel interface for a tag
// transfers ownership
MetaGraphDef* loadMetaGraph(const std::string& exportDir, bool multiThreaded = false,
    const std::string& tag = kSavedModelTagServe);

// return a new, empty session
// transfers ownership
Session* createSession(bool multiThreaded = false);

// return a new session that contains an already loaded meta graph whose exportDir must be given in
// order to load and initialize the variables within the session
// transfers ownership
Session* createSession(MetaGraphDef* metaGraph, const std::string& exportDir,
    bool multiThreaded = false);

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

// use a namespace alias
namespace tf = tensorflow;

#endif // PHYSICSTOOLS_TENSORFLOW_TENSORFLOW_H
