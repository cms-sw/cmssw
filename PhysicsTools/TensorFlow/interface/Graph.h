/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_GRAPH_H
#define PHYSICSTOOLS_TENSORFLOW_GRAPH_H

#include <tensorflow/c/c_api.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace tf
{

// the Graph class
class Graph
{
public:
    // default constructor
    Graph();

    // constructor with initialization
    Graph(const std::string& exportDir, const std::string& tag = "serve");

    // disable implicit copy constructor
    Graph(const Graph&) = delete;

    // destructor
    ~Graph();

    // initialize the tensorflow graph and session objects
    void init(const std::string& exportDir, const std::string& tag = "serve");

    // reset the tensorflow graph and session objects as well as input/output vectors
    void reset();

    // returns true if the tensorflow graph object is not initialized yet, false otherwise
    inline bool empty() const
    {
        return tf_graph_ == nullptr;
    }

    // returns the pointer to the tensorflow graph object
    inline TF_Graph* getTFGraph()
    {
        return tf_graph_;
    }

    // returns the exportDir that was used to create the current tensorflow graph object
    inline std::string getExportDir() const
    {
        return exportDir_;
    }

    // returns the pointer to a tensorflow operation object defined by its name, ownership is not
    // transferred
    TF_Operation* getTFOperation(const std::string& name);

private:
    // pointer to the tensorflow graph object
    TF_Graph* tf_graph_;

    // the exportDir of the current graph
    std::string exportDir_;
};

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_GRAPH_H
