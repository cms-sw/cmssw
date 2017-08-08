/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_GRAPH_H
#define PHYSICSTOOLS_TENSORFLOW_GRAPH_H

#include <algorithm>
#include <stdexcept>
#include <string>

#include <tensorflow/c/c_api.h>

#include "PhysicsTools/TensorFlow/interface/Tensor.h"

namespace tf
{

// forward declaration
class Graph;

// generic class containing all information of inputs to / ouptuts from a graph
class GraphIO
{
public:
    // constructur
    GraphIO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex = 0);

    // destructor
    virtual ~GraphIO();

private:
    Tensor* tensor;
    std::string opName;
    int opIndex;

    TF_Output tf_output;

    friend Graph;
};

// the Graph class
class Graph
{
public:
    // constructor
    Graph(const std::string& exportDir = "", const std::string& tag = "serve");

    // disable implicit copy constructor
    Graph(const Graph& g) = delete;

    // destructor
    virtual ~Graph();

    // initialize the tensorflow graph and session objects
    void init(const std::string& exportDir, const std::string& tag = "serve");

    // reset the tensorflow graph and session objects as well as input/output vectors
    void reset();

    // returns true if the tensorflow graph object is not initialized yet, false otherwise
    inline bool empty() const
    {
        return tf_graph == 0;
    }

    // defines an input to the graph that is connected to an operation with name and index
    GraphIO* defineInput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // defines an output to the graph that is connected to an operation with name and index
    GraphIO* defineOutput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously defined input from an operation with name and index
    void removeInput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously defined input
    void removeInput(GraphIO* input);

    // removes a previously defined output from an operation with name and index
    void removeOutput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously defined output
    void removeOutput(GraphIO* input);

    // returns true if a tensor was previously defined as an input to an operation with name and
    // index, false otherwise
    bool hasInput(Tensor* tensor, const std::string& opName, int opIndex = 0) const;

    // retrns true if a tensor was previously defined as an input py passing the input object, false
    // otherwise
    bool hasInput(GraphIO* input) const;

    // returns true if a tensor was previously defined as an output of an operation with name and
    // index, false otherwise
    bool hasOutput(Tensor* tensor, const std::string& opName, int opIndex = 0) const;

    // retrns true if a tensor was previously defined as an output py passing the output object,
    // false otherwise
    bool hasOutput(GraphIO* output) const;

    // returns the number of inputs
    inline size_t nInputs() const
    {
        return inputs.size();
    }

    // returns the number of outputs
    inline size_t nOutputs() const
    {
        return outputs.size();
    }

    // performs a graph evaluation with all previously defined inputs and outputs
    void eval();

private:
    bool preparedEval;

    // pointers to the tensorflow graph and session objects
    TF_Graph* tf_graph;
    TF_Session* tf_session;

    // vectors of input and output objects
    std::vector<GraphIO*> inputs;
    std::vector<GraphIO*> outputs;

    // vectors for caching objects required for evaluation
    std::vector<TF_Output> inputOutputs;
    std::vector<TF_Output> outputOutputs;
    std::vector<TF_Tensor*> inputTensors;
    std::vector<TF_Tensor*> outputTensors;

    // prepares the cache vectors that are required for the evaluation
    void prepareEval();
};

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_GRAPH_H
