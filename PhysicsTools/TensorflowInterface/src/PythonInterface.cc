/*
 * Python interface.
 *
 * Author:
 *   Marcel Rieger
 */

#include "PhysicsTools/TensorflowInterface/interface/PythonInterface.h"

namespace tf
{

PythonInterface::PythonInterface()
    : logCategory("PythonInterface")
    , context(0)
{
    initialize();
    startContext();
}

PythonInterface::~PythonInterface()
{
    release(context);
    finalize();
}

void PythonInterface::except(PyObject* obj, const std::string& msg) const
{
    // in Python we know that an error occured when an object is NULL
    if (obj == NULL)
    {
        // check if there is a python error on the stack
        if (PyErr_Occurred() != NULL)
        {
            PyErr_PrintEx(0);
        }
        throw std::runtime_error("a python error occured: " + msg);
    }
}

void PythonInterface::release(PyObject*& ptr) const
{
    Py_XDECREF(ptr);
    ptr = 0;
}

PyObject* PythonInterface::get(const std::string& name) const
{
    if (!hasContext())
    {
        throw std::runtime_error("python context not yet started");
    }

    PyObject* obj = PyDict_GetItemString(context, name.c_str());
    except(obj, "could not get object '" + name + "'");

    return obj;
}

PyObject* PythonInterface::call(PyObject* callable, PyObject* args) const
{
    // check if args is a tuple
    size_t nArgs = 0;
    if (args != NULL)
    {
        if (!PyTuple_Check(args))
        {
            throw std::runtime_error("args for function call is not a tuple");
        }
        nArgs = PyTuple_Size(args);
    }

    LogDebug(logCategory) << "invoke callable with " << nArgs << " argument(s)";

    // simply call the callable with args and check for errors afterwards
    PyObject* result = PyObject_CallObject(callable, args);
    except(result, "error during invocation of callable");

    return result;
}

PyObject* PythonInterface::call(const std::string& name, PyObject* args) const
{
    LogDebug(logCategory) << "invoke callable " << name;

    PyObject* callable = get(name);
    PyObject* result = call(callable, args);

    return result;
}

PyObject* PythonInterface::call(const std::string& name, int arg) const
{
    PyObject* args = Py_BuildValue("(i)", arg);
    PyObject* result = call(name, args);
    release(args);

    return result;
}

PyObject* PythonInterface::call(const std::string& name, double arg) const
{
    PyObject* args = Py_BuildValue("(f)", arg);
    PyObject* result = call(name, args);
    release(args);

    return result;
}

PyObject* PythonInterface::call(const std::string& name, const std::string& arg) const
{
    PyObject* args = Py_BuildValue("(s)", arg.c_str());
    PyObject* result = call(name, args);
    release(args);

    return result;
}

PyObject* PythonInterface::createTuple(const std::vector<int>& v) const
{
    PyObject* tpl = PyTuple_New(v.size());

    for (size_t i = 0; i < v.size(); i++)
    {
        PyTuple_SetItem(tpl, i, PyInt_FromLong(v[i]));
    }

    return tpl;
}

PyObject* PythonInterface::createTuple(const std::vector<double>& v) const
{
    PyObject* tpl = PyTuple_New(v.size());

    for (size_t i = 0; i < v.size(); i++)
    {
        PyTuple_SetItem(tpl, i, PyFloat_FromDouble(v[i]));
    }

    return tpl;
}

PyObject* PythonInterface::createTuple(const std::vector<std::string>& v) const
{
    PyObject* tpl = PyTuple_New(v.size());

    for (size_t i = 0; i < v.size(); i++)
    {
        PyTuple_SetItem(tpl, i, PyString_FromString(v[i].c_str()));
    }

    return tpl;
}

void PythonInterface::runScript(const std::string& script)
{
    edm::LogInfo(logCategory) << "run script";

    if (!hasContext())
    {
        throw std::runtime_error("python context not yet started");
    }

    // run the script in our context
    PyObject* result = PyRun_String(script.c_str(), Py_file_input, context, context);
    except(result, "error during execution of script");
    release(result);
}

void PythonInterface::runFile(const std::string& filename)
{
    edm::LogInfo(logCategory) << "run file from " << filename;

    // read the content of the file
    std::ifstream ifs(filename);
    std::string script;
    script.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    // run the script
    runScript(script);
}

void PythonInterface::initialize() const
{
    edm::LogInfo(logCategory) << "initialize";

    if (nConsumers == 0 && !Py_IsInitialized())
    {
        PyEval_InitThreads();
        Py_Initialize();
    }
    nConsumers++;

    LogDebug(logCategory) << nConsumers << " consumers";
}

void PythonInterface::finalize() const
{
    edm::LogInfo(logCategory) << "finalize";

    if (nConsumers == 1 && Py_IsInitialized())
    {
        Py_Finalize();
    }
    if (nConsumers != 0)
    {
        nConsumers--;
    }
    else
    {
        edm::LogWarning(logCategory) << "number of consumers cannot be negative";
    }

    LogDebug(logCategory) << nConsumers << " consumers";
}

bool PythonInterface::hasContext() const
{
    return context != 0;
}

void PythonInterface::startContext()
{
    edm::LogInfo(logCategory) << "start context";

    if (hasContext())
    {
        throw std::runtime_error("python context already started");
    }

    // create the main module
    PyObject* main = PyImport_AddModule("__main__");

    // define the globals of the main module as our context
    context = PyModule_GetDict(main);

    // since PyModule_GetDict returns a borrowed reference, increase the count to own one
    Py_INCREF(context);
}

size_t PythonInterface::nConsumers = 0;

} // namespace tf
