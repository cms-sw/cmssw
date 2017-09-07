/*
 * TensorFlow session interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_SESSION_H
#define PHYSICSTOOLS_TENSORFLOW_SESSION_H

#include <algorithm>
#include <vector>
#include <cstring>

#include <tensorflow/c/c_api.h>

#include "PhysicsTools/TensorFlow/interface/IO.h"
#include "PhysicsTools/TensorFlow/interface/Tensor.h"
#include "PhysicsTools/TensorFlow/interface/Graph.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace tf
{

// the Session class
class Session
{
public:
    // default constructor
    Session();

    // constructor with a graph, ownership is not transferred
    Session(Graph* graph, bool threads = false);

    // disable implicit copy constructor
    Session(const Session&) = delete;

    // destructor
    ~Session();

    // initialize the tensorflow session object with a graph, ownership is not transferred, returns
    // true when the variable initialization succeeded, false otherwise
    bool init(Graph* graph, bool threads = false);

    // reset the tensorflow session and default graph objects
    void reset();

    // returns true if the tensorflow session object is not initialized yet, false otherwise
    inline bool empty() const
    {
        return tf_session_ == nullptr;
    }

    // initializes all variables using configurable initializer ops that default to the SavedModel
    // values, returns true when the ops were found, false otherwise
    bool initVariables(const std::string& restoreOpName = "save/restore_all",
        const std::string& varOpName = "save/Const");

    // creates an input/output object that is connected to an operation with name and index, but
    // does not add it yet, ownership is transferred to the caller
    IO* createIO(Tensor* tensor, const std::string& opName, int opIndex = 0) const;

    // adds an input to the graph that is connected to an operation with name and index
    IO* addInput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // adds an output to the graph that is connected to an operation with name and index
    IO* addOutput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously added input from an operation with name and index
    void removeInput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously added input
    void removeInput(IO* input);

    // removes a previously added output from an operation with name and index
    void removeOutput(Tensor* tensor, const std::string& opName, int opIndex = 0);

    // removes a previously added output
    void removeOutput(IO* input);

    // returns true if a tensor was previously added as an input to an operation with name and
    // index, false otherwise
    bool hasInput(Tensor* tensor, const std::string& opName, int opIndex = 0) const;

    // retrns true if a tensor was previously added as an input py passing the input object, false
    // otherwise
    bool hasInput(IO* input) const;

    // returns true if a tensor was previously added as an output of an operation with name and
    // index, false otherwise
    bool hasOutput(Tensor* tensor, const std::string& opName, int opIndex = 0) const;

    // retrns true if a tensor was previously added as an output py passing the output object,
    // false otherwise
    bool hasOutput(IO* output) const;

    // returns the number of previously added inputs
    inline size_t nInputs() const
    {
        return inputs_.size();
    }

    // returns the number of previously added outputs
    inline size_t nOutputs() const
    {
        return outputs_.size();
    }

    // run a stateful evaluation of the default graph with all previously added inputs and outputs
    void run();

    // run a stateless evaluation of the default graph with inputs and outputs passed as arguments
    void run(const IOs inputs, const IOs& outputs) const;

private:
    // flag that denotes if the stateful evaluation is prepared
    bool prepared_;

    // pointer to the tensorflow session object
    TF_Session* tf_session_;

    // pointer to the default graph
    Graph* graph_;

    // vectors of input and output objects
    IOs inputs_;
    IOs outputs_;

    // vectors for caching objects required for evaluation
    std::vector<TF_Output> inputOutputs_;
    std::vector<TF_Output> outputOutputs_;
    std::vector<TF_Tensor*> inputTensors_;
    std::vector<TF_Tensor*> outputTensors_;

    // prepares the cache vectors that are required for the stateful evaluation
    void prepare();
};

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_SESSION_H
