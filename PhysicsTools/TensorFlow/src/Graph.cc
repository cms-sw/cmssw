/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <iostream>
#include "PhysicsTools/TensorFlow/interface/Graph.h"

namespace tf
{

GraphIO::GraphIO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex)
    : tensor(tensor)
    , opName(opName)
    , opIndex(opIndex)
{
    // create the tf_output object, i.e. a struct of op and index
    tf_output.oper = tf_operation;
    tf_output.index = opIndex;
}

GraphIO::~GraphIO()
{
}

Graph::Graph()
    : preparedEval(false)
    , tf_graph(0)
    , tf_session(0)
{
}

Graph::Graph(const std::string& filename, const std::string& tag)
    : preparedEval(false)
    , tf_graph(0)
    , tf_session(0)
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
    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    const char* tags[] = { tag.c_str() };

    // apply all session options
    for (size_t i = 0, s = sessionOptions_.size(); i < s; i++)
    {
        TF_SetTarget(tf_sessionOptions, sessionOptions_[i].c_str());
    }

    // initialize an empty graph
    tf_graph = TF_NewGraph();

    TF_Status* status = TF_NewStatus();

    // create the session and load everything into tf_graph
    tf_session = TF_LoadSessionFromSavedModel(
        sessionOptions, 0, exportDir.c_str(), tags, 1, tf_graph, 0, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw std::runtime_error("error while loading graph: " + std::string(TF_Message(status)));
    }

    // some cleanup
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(sessionOptions);
}

void Graph::reset()
{
    preparedEval = false;

    // clear all inputs
    while (nInputs() > 0)
    {
        removeInput(inputs[0]);
    }

    // clear all outputs
    while (nOutputs() > 0)
    {
        removeOutput(outputs[0]);
    }

    // close and delete the session object
    if (tf_session)
    {
        TF_Status* status = TF_NewStatus();

        TF_CloseSession(tf_session, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw std::runtime_error("error while closing session: "
                + std::string(TF_Message(status)));
        }

        TF_DeleteSession(tf_session, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw std::runtime_error("error while deleting session: "
                + std::string(TF_Message(status)));
        }
        tf_session = 0;

        TF_DeleteStatus(status);
    }

    // delete the graph object
    if (tf_graph)
    {
        TF_DeleteGraph(tf_graph);
        tf_graph = 0;
    }
}

GraphIO* Graph::defineInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // the tensor must be initialized
    if (tensor->empty())
    {
        throw std::runtime_error("cannot define uninitialized input tensor for operation: "
            + opName + " " + std::to_string(opIndex));
    }

    // check for duplicate
    if (hasInput(tensor, opName, opIndex))
    {
        throw std::runtime_error("duplicate input tensor defined for operation: " + opName
            + " " + std::to_string(opIndex));
    }

    // get a pointer to the associated operation
    TF_Operation* operation = TF_GraphOperationByName(tf_graph, opName.c_str());
    if (!operation)
    {
        throw std::runtime_error("no such input operation in graph: " + opName);
    }

    // create and store the input object
    GraphIO* input = new GraphIO(tensor, operation, opName, opIndex);
    inputs.push_back(input);

    preparedEval = false;

    return input;
}

GraphIO* Graph::defineOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // check for duplicate
    if (hasOutput(tensor, opName, opIndex))
    {
        throw std::runtime_error("duplicate output tensor defined for operation: " + opName
            + " " + std::to_string(opIndex));
    }

    // get a pointer to the associated operation
    TF_Operation* operation = TF_GraphOperationByName(tf_graph, opName.c_str());
    if (!operation)
    {
        throw std::runtime_error("no such output operation in graph: " + opName);
    }

    // create and store the output object
    GraphIO* output = new GraphIO(tensor, operation, opName, opIndex);
    outputs.push_back(output);

    preparedEval = false;

    return output;
}

void Graph::removeInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    std::vector<GraphIO*>::iterator it;
    for (it = inputs.begin(); it != inputs.end(); it++)
    {
        if ((*it)->getTensor() == tensor && (*it)->getOpName() == opName
            && (*it)->getOpIndex() == opIndex)
        {
            delete *it;
            inputs.erase(it);
            preparedEval = false;
            break;
        }
    }
}

void Graph::removeInput(GraphIO* input)
{
    std::vector<GraphIO*>::iterator it = std::find(inputs.begin(), inputs.end(), input);
    if (it != inputs.end())
    {
        delete *it;
        inputs.erase(it);
        preparedEval = false;
    }
}

void Graph::removeOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    std::vector<GraphIO*>::iterator it;
    for (it = outputs.begin(); it != outputs.end(); it++)
    {
        if ((*it)->getTensor() == tensor && (*it)->getOpName() == opName
            && (*it)->getOpIndex() == opIndex)
        {
            delete *it;
            outputs.erase(it);
            preparedEval = false;
            break;
        }
    }
}

void Graph::removeOutput(GraphIO* output)
{
    std::vector<GraphIO*>::iterator it = std::find(outputs.begin(), outputs.end(), output);
    if (it != outputs.end())
    {
        delete *it;
        outputs.erase(it);
        preparedEval = false;
    }
}

bool Graph::hasInput(GraphIO* input) const
{
    return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
}

bool Graph::hasOutput(GraphIO* output) const
{
    return std::find(outputs.begin(), outputs.end(), output) != outputs.end();
}

bool Graph::hasInput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    std::vector<GraphIO*>::const_iterator it;
    for (it = inputs.begin(); it != inputs.end(); it++)
    {
        if ((*it)->getTensor() == tensor && (*it)->getOpName() == opName
            && (*it)->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

bool Graph::hasOutput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    std::vector<GraphIO*>::const_iterator it;
    for (it = outputs.begin(); it != outputs.end(); it++)
    {
        if ((*it)->getTensor() == tensor && (*it)->getOpName() == opName
            && (*it)->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

void Graph::eval()
{
    if (!tf_session)
    {
        throw std::runtime_error("cannot evaluate uninitialized graph");
    }

    // prepare the evaluation objects
    prepareEval();

    size_t nIn = nInputs();
    size_t nOut = nOutputs();

    // clear previous outputs
    for (size_t i = 0; i < nOut; ++i)
    {
        outputs[i]->getTensor()->reset();
    }
    outputTensors.clear();
    outputTensors.resize(nOut, 0);

    // actual evaluation
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(
        tf_session,
        0, // run options
        nIn == 0 ? 0 : &inputOutputs[0], nIn == 0 ? 0 : &inputTensors[0], nIn,
        nOut == 0 ? 0 : &outputOutputs[0], nOut == 0 ? 0 : &outputTensors[0], nOut,
        0, 0, // target ops, number of targets
        0, // run metadata
        status);

    // check the status
    if (TF_GetCode(status) != TF_OK)
    {
        throw std::runtime_error("error while evaluating graph: "
            + std::string(TF_Message(status)));
    }

    // sync outputs again
    for (size_t i = 0; i < nOut; ++i)
    {
        outputs[i]->getTensor()->init(outputTensors[i]);
    }

    // cleanup
    TF_DeleteStatus(status);
}

void Graph::prepareEval()
{
    if (preparedEval)
    {
        return;
    }

    // clear input objects and set them again
    inputOutputs.clear();
    inputTensors.clear();
    std::vector<GraphIO*>::iterator it;
    for (it = inputs.begin(); it != inputs.end(); it++)
    {
        inputOutputs.push_back((*it)->getTFOutput());
        inputTensors.push_back((*it)->getTensor()->getTFTensor());
    }

    // clear output objects and set them again
    outputOutputs.clear();
    outputTensors.clear();
    for (it = outputs.begin(); it != outputs.end(); it++)
    {
        outputOutputs.push_back((*it)->getTFOutput());
        outputTensors.push_back((*it)->getTensor()->getTFTensor());
    }

    preparedEval = true;
}

} // namespace tf
