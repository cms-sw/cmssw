/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "PhysicsTools/TensorFlow/interface/Graph.h"

namespace tf
{

GraphIO::GraphIO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex)
    : tensor_(tensor)
    , opName_(opName)
    , opIndex_(opIndex)
{
    // create the tf_output object, i.e. a struct of op and index
    tf_output_.oper = tf_operation;
    tf_output_.index = opIndex;
}

GraphIO::~GraphIO()
{
}

Graph::Graph()
    : preparedEval_(false)
    , tf_graph_(nullptr)
    , tf_session_(nullptr)
{
}


Graph::Graph(const std::string& filename, const std::string& tag)
    : preparedEval_(false)
    , tf_graph_(nullptr)
    , tf_session_(nullptr)
{
    init(filename, tag);
}

Graph::~Graph()
{
    reset();
}

void Graph::init(const std::string& exportDir, const std::string& tag)
{
    reset();

    // disable tensorflow logging by default
    setenv("TF_CPP_MIN_LOG_LEVEL", "3", 0);

    // config objects
    TF_SessionOptions* tf_sessionOptions = TF_NewSessionOptions();
    const char* tags[] = { tag.c_str() };

    // apply all session options
    for (size_t i = 0, s = sessionOptions_.size(); i < s; i++)
    {
        TF_SetTarget(tf_sessionOptions, sessionOptions_[i].c_str());
    }

    // initialize an empty graph
    tf_graph_ = TF_NewGraph();

    TF_Status* status = TF_NewStatus();

    // create the session and load everything into tf_graph
    tf_session_ = TF_LoadSessionFromSavedModel(
        tf_sessionOptions, nullptr, exportDir.c_str(), tags, 1, tf_graph_, nullptr, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidSession") << "error while loading graph: "
            << TF_Message(status);
    }

    // some cleanup
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(tf_sessionOptions);
}

void Graph::reset()
{
    preparedEval_ = false;

    // clear all inputs
    while (nInputs() > 0)
    {
        removeInput(inputs_[0]);
    }

    // clear all outputs
    while (nOutputs() > 0)
    {
        removeOutput(outputs_[0]);
    }

    // close and delete the session object
    if (tf_session_)
    {
        TF_Status* status = TF_NewStatus();

        TF_CloseSession(tf_session_, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw cms::Exception("InvalidSession") << "error while closing session: "
                << TF_Message(status);
        }

        TF_DeleteSession(tf_session_, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw cms::Exception("InvalidSession") << "error while deleting session: "
                << TF_Message(status);
        }
        tf_session_ = nullptr;

        TF_DeleteStatus(status);
    }

    // delete the graph object
    if (tf_graph_)
    {
        TF_DeleteGraph(tf_graph_);
        tf_graph_ = nullptr;
    }
}

GraphIO* Graph::defineInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // the tensor must be initialized
    if (tensor->empty())
    {
        throw cms::Exception("InvalidTensor")
            << "cannot define uninitialized input tensor for operation: " << opName << " "
            << opIndex;
    }

    // check for duplicate
    if (hasInput(tensor, opName, opIndex))
    {
        throw cms::Exception("InvalidInput") << "duplicate input tensor defined for operation: "
            << opName << " " << opIndex;
    }

    // get a pointer to the associated operation
    TF_Operation* operation = TF_GraphOperationByName(tf_graph_, opName.c_str());
    if (!operation)
    {
        throw cms::Exception("InvalidOperation") << "no such input operation in graph: " << opName;
    }

    // create and store the input object
    GraphIO* input = new GraphIO(tensor, operation, opName, opIndex);
    inputs_.push_back(input);

    preparedEval_ = false;

    return input;
}

GraphIO* Graph::defineOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // check for duplicate
    if (hasOutput(tensor, opName, opIndex))
    {
        throw cms::Exception("InvalidOutput") << "duplicate output tensor defined for operation: "
            << opName << " " << opIndex;
    }

    // get a pointer to the associated operation
    TF_Operation* operation = TF_GraphOperationByName(tf_graph_, opName.c_str());
    if (!operation)
    {
        throw cms::Exception("InvalidOperation") << "no such input operation in graph: " << opName;
    }

    // create and store the output object
    GraphIO* output = new GraphIO(tensor, operation, opName, opIndex);
    outputs_.push_back(output);

    preparedEval_ = false;

    return output;
}

void Graph::removeInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    for (size_t i = 0, n = nInputs(); i < n; i++)
    {
        if (inputs_[i]->getTensor() == tensor && inputs_[i]->getOpName() == opName
            && inputs_[i]->getOpIndex() == opIndex)
        {
            delete inputs_[i];
            inputs_.erase(inputs_.begin() + i);
            preparedEval_ = false;
            break;
        }
    }
}

void Graph::removeInput(GraphIO* input)
{
    std::vector<GraphIO*>::iterator it = std::find(inputs_.begin(), inputs_.end(), input);
    if (it != inputs_.end())
    {
        delete *it;
        inputs_.erase(it);
        preparedEval_ = false;
    }
}

void Graph::removeOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        if (outputs_[i]->getTensor() == tensor && outputs_[i]->getOpName() == opName
            && outputs_[i]->getOpIndex() == opIndex)
        {
            delete outputs_[i];
            outputs_.erase(outputs_.begin() + i);
            preparedEval_ = false;
            break;
        }
    }
}

void Graph::removeOutput(GraphIO* output)
{
    std::vector<GraphIO*>::iterator it = std::find(outputs_.begin(), outputs_.end(), output);
    if (it != outputs_.end())
    {
        delete *it;
        outputs_.erase(it);
        preparedEval_ = false;
    }
}

bool Graph::hasInput(GraphIO* input) const
{
    return std::find(inputs_.begin(), inputs_.end(), input) != inputs_.end();
}

bool Graph::hasOutput(GraphIO* output) const
{
    return std::find(outputs_.begin(), outputs_.end(), output) != outputs_.end();
}

bool Graph::hasInput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    for (size_t i = 0, n = nInputs(); i < n; i++)
    {
        if (inputs_[i]->getTensor() == tensor && inputs_[i]->getOpName() == opName
            && inputs_[i]->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

bool Graph::hasOutput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        if (outputs_[i]->getTensor() == tensor && outputs_[i]->getOpName() == opName
            && outputs_[i]->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

void Graph::eval()
{
    if (!tf_session_)
    {
        throw cms::Exception("InvalidGraph") << "cannot evaluate uninitialized graph";
    }

    // prepare the evaluation objects
    prepareEval();

    size_t nIn = nInputs();
    size_t nOut = nOutputs();

    // clear previous outputs
    for (size_t i = 0; i < nOut; ++i)
    {
        outputs_[i]->getTensor()->reset();
    }
    outputTensors_.clear();
    outputTensors_.resize(nOut, nullptr);

    // actual evaluation
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(
        tf_session_,
        nullptr, // run options
        nIn == 0 ? nullptr : &inputOutputs_[0], nIn == 0 ? nullptr : &inputTensors_[0], nIn,
        nOut == 0 ? nullptr : &outputOutputs_[0], nOut == 0 ? nullptr : &outputTensors_[0], nOut,
        nullptr, 0, // target ops, number of targets
        nullptr, // run metadata
        status);

    // check the status
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidGraph") << "error while evaluating graph: "
            << (TF_Message(status));
    }

    // sync outputs again
    for (size_t i = 0; i < nOut; ++i)
    {
        outputs_[i]->getTensor()->init(outputTensors_[i]);
    }

    // cleanup
    TF_DeleteStatus(status);
}

void Graph::prepareEval()
{
    if (preparedEval_)
    {
        return;
    }

    // clear input objects and set them again
    inputOutputs_.clear();
    inputTensors_.clear();
    std::vector<GraphIO*>::iterator it;
    for (size_t i = 0, n = nInputs(); i < n; i++) {
        inputOutputs_.push_back(inputs_[i]->getTFOutput());
        inputTensors_.push_back(inputs_[i]->getTensor()->getTFTensor());
    }

    // clear output objects and set them again
    outputOutputs_.clear();
    outputTensors_.clear();
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        outputOutputs_.push_back(outputs_[i]->getTFOutput());
        outputTensors_.push_back(outputs_[i]->getTensor()->getTFTensor());
    }

    preparedEval_ = true;
}

} // namespace tf
