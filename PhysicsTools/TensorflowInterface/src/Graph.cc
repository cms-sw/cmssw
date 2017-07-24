/*
 * Generic Tensorflow graph representation.
 *
 * Author:
 *   Marcel Rieger
 */

#include "PhysicsTools/TensorflowInterface/interface/Graph.h"

namespace tf
{

Graph::Graph(const std::string& filename)
    : logCategory("TensorflowGraph")
    , python()
    , pyInputs(0)
    , pyOutputs(0)
    , pyEvalSession(0)
{
    LogDebug(logCategory) << "initialize graph";

    python.runScript(embeddedTensorflowScript);

    // import tensorflow
    PyObject* result = python.call("import_tf");
    python.release(result);

    // create the in/output dicts
    pyInputs = PyDict_New();
    pyOutputs = PyDict_New();

    if (!filename.empty())
    {
        load(filename);
    }
}

Graph::~Graph()
{
    // cleanup python objects
    python.release(pyInputs);
    python.release(pyOutputs);
}

Tensor* Graph::defineInput(Tensor* tensor)
{
    if (!tensor || tensor->getName().empty())
    {
        throw std::runtime_error("input tensor must not be empty or unnamed");
    }

    removeInput(tensor->getName());
    inputs[tensor->getName()] = tensor;

    return tensor;
}

Tensor* Graph::defineOutput(Tensor* tensor)
{
    if (!tensor || tensor->getName().empty())
    {
        throw std::runtime_error("output tensor must not be empty or unnamed");
    }

    removeOutput(tensor->getName());
    outputs[tensor->getName()] = tensor;

    return tensor;
}

void Graph::removeInput(const std::string& name)
{
    if (hasInput(name))
    {
        inputs.erase(name);
    }
}

void Graph::removeOutput(const std::string& name)
{
    if (hasOutput(name))
    {
        outputs.erase(name);
    }
}

bool Graph::hasInput(const std::string& name) const
{
    return inputs.find(name) != inputs.end();
}

bool Graph::hasOutput(const std::string& name) const
{
    return outputs.find(name) != outputs.end();
}

Tensor* Graph::getInput(const std::string& name)
{
    return hasInput(name) ? inputs.find(name)->second : 0;
}

Tensor* Graph::getOutput(const std::string& name)
{
    return hasOutput(name) ? outputs.find(name)->second : 0;
}

void Graph::load(const std::string& filename)
{
    edm::LogInfo(logCategory) << "load graph from " << filename;

    PyObject* result = python.call("start_session");
    python.release(result);

    result = python.call("load_graph", filename);
    python.release(result);

    pyEvalSession = python.get("eval_session");
}

void Graph::eval()
{
    LogDebug(logCategory) << "evaluate";

    if (!pyEvalSession)
    {
        throw std::runtime_error("cannot eval session, graph not loaded yet");
    }

    // sync arrays of input tensors
    std::map<std::string, Tensor*>::iterator it;
    for (it = inputs.begin(); it != inputs.end(); it++)
    {
        if (it->second->isEmpty())
        {
            throw std::runtime_error("cannot set non-initialized tensor as input");
        }
        PyDict_SetItemString(pyInputs, it->first.c_str(), (PyObject*)it->second->getArray());
    }

    // clear arrays of output tensors from last call
    for (it = outputs.begin(); it != outputs.end(); it++)
    {
        it->second->setArray((PyObject*)0);
        PyDict_SetItemString(pyOutputs, it->first.c_str(), Py_None);
    }

    // actual evaluation
    PyObject* pyEvalArgs = PyTuple_New(2);
    Py_INCREF(pyInputs);
    PyTuple_SetItem(pyEvalArgs, 0, pyInputs);
    Py_INCREF(pyOutputs);
    PyTuple_SetItem(pyEvalArgs, 1, pyOutputs);
    python.call(pyEvalSession, pyEvalArgs);
    Py_DECREF(pyEvalArgs);

    // update arrays of output tensors
    for (it = outputs.begin(); it != outputs.end(); it++)
    {
        PyObject* obj = PyDict_GetItemString(pyOutputs, it->first.c_str());
        python.except(obj, "evaluation for tensor '" + it->first + "' failed");

        Py_INCREF(obj);
        it->second->setArray(obj);
    }
}

PythonInterface& Graph::getPythonInterface()
{
    return python;
}

} // namespace tf
