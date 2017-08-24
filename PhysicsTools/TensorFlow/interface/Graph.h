/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_GRAPH_H
#define PHYSICSTOOLS_TENSORFLOW_GRAPH_H

#include <algorithm>
#include <string>

#include <tensorflow/c/c_api.h>

#include "PhysicsTools/TensorFlow/interface/Tensor.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace tf
{

// generic class containing all information of inputs to / ouptuts from a graph
class GraphIO
{
public:
    // constructur
    GraphIO(Tensor* tensor, TF_Operation* tf_operation, const std::string& opName, int opIndex = 0);

    // destructor
    ~GraphIO();

    // returns the pointer to the tensor instance
    inline Tensor* getTensor()
    {
        return tensor_;
    }

    // returns a reference to the tensorflow output object
    inline TF_Output& getTFOutput()
    {
        return tf_output_;
    }

    // returns the operation name
    inline std::string& getOpName()
    {
        return opName_;
    }

    // returns the operation index
    inline int getOpIndex()
    {
        return opIndex_;
    }

private:
    Tensor* tensor_;
    std::string opName_;
    int opIndex_;

    TF_Output tf_output_;
};

// the Graph class
class Graph
{
public:
    // default constructor
    Graph();

    // constructor with initialization
    Graph(const std::string& exportDir, const std::string& tag = "serve");

    // disable implicit copy constructor
    Graph(const Graph& g) = delete;

    // destructor
    ~Graph();

    // initialize the tensorflow graph and session objects
    void init(const std::string& exportDir, const std::string& tag = "serve");

    // reset the tensorflow graph and session objects as well as input/output vectors, but not the
    // session options
    void reset();

    // adds a session option of the format "key:value"
    inline void addSessionOption(const std::string& opt)
    {
        sessionOptions_.push_back(opt);
    }

    // clears all session options
    inline void clearSessionOptions()
    {
        sessionOptions_.clear();
    }

    // returns true if the tensorflow graph object is not initialized yet, false otherwise
    inline bool empty() const
    {
        return tf_graph_ == nullptr;
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
        return inputs_.size();
    }

    // returns the number of outputs
    inline size_t nOutputs() const
    {
        return outputs_.size();
    }

    // performs a graph evaluation with all previously defined inputs and outputs
    void eval();

private:
    bool preparedEval_;

    // session option strings of the format "key:value"
    std::vector<std::string> sessionOptions_;

    // pointers to the tensorflow graph and session objects
    TF_Graph* tf_graph_;
    TF_Session* tf_session_;

    // vectors of input and output objects
    std::vector<GraphIO*> inputs_;
    std::vector<GraphIO*> outputs_;

    // vectors for caching objects required for evaluation
    std::vector<TF_Output> inputOutputs_;
    std::vector<TF_Output> outputOutputs_;
    std::vector<TF_Tensor*> inputTensors_;
    std::vector<TF_Tensor*> outputTensors_;

    // prepares the cache vectors that are required for the evaluation
    void prepareEval();
};

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_GRAPH_H
