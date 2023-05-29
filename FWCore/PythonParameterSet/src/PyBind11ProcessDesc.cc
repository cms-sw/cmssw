#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/PythonParameterSet/src/initializePyBind11Module.h"
#include "FWCore/PythonParameterSet/interface/PyBind11Wrapper.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <iostream>

PyBind11InterpreterSentry::PyBind11InterpreterSentry(bool ownsInterpreter)
    : mainModule(), ownsInterpreter_(ownsInterpreter) {
  if (ownsInterpreter_) {
    pybind11::initialize_interpreter();
  }
}

PyBind11InterpreterSentry::~PyBind11InterpreterSentry() {
  if (ownsInterpreter_) {
    mainModule = pybind11::object();
    pybind11::finalize_interpreter();
  }
}

PyBind11ProcessDesc::PyBind11ProcessDesc() : theProcessPSet(), theInterpreter(false) {}

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config) : theProcessPSet(), theInterpreter(true) {
  edm::python::initializePyBind11Module();
  prepareToRead();
  read(config);
}

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config, int argc, char* argv[])
    : theProcessPSet(),
      theInterpreter(true)

{
  edm::python::initializePyBind11Module();
  prepareToRead();
  {
#if PY_MAJOR_VERSION >= 3
    typedef std::unique_ptr<wchar_t[], decltype(&PyMem_RawFree)> WArgUPtr;
    std::vector<WArgUPtr> v_argv;
    std::vector<wchar_t*> vp_argv;
    v_argv.reserve(argc);
    vp_argv.reserve(argc);
    for (int i = 0; i < argc; i++) {
      v_argv.emplace_back(Py_DecodeLocale(argv[i], nullptr), &PyMem_RawFree);
      vp_argv.emplace_back(v_argv.back().get());
    }

    wchar_t** argvt = vp_argv.data();
#else
    char** argvt = argv;
#endif

    PySys_SetArgv(argc, argvt);
  }
  read(config);
}

PyBind11ProcessDesc::~PyBind11ProcessDesc() = default;

void PyBind11ProcessDesc::prepareToRead() {
  //  pybind11::scoped_interpreter guard{};
  theInterpreter.mainModule = pybind11::module::import("__main__");
  theInterpreter.mainModule.attr("processDesc") = this;
  theInterpreter.mainModule.attr("processPSet") = &theProcessPSet;
}

void PyBind11ProcessDesc::read(std::string const& config) {
  try {
    // if it ends with py, it's a file
    if (config.substr(config.size() - 3) == ".py") {
      readFile(config);
    } else {
      readString(config);
    }
  } catch (pybind11::error_already_set const& e) {
    edm::pythonToCppException("Configuration", e.what());
  }
}

void PyBind11ProcessDesc::readFile(std::string const& fileName) {
  pybind11::eval_file(fileName);
  std::string command("process.fillProcessDesc(processPSet)");
  pybind11::exec(command);
}

void PyBind11ProcessDesc::readString(std::string const& pyConfig) {
  std::string command = pyConfig;
  command += "\nprocess.fillProcessDesc(processPSet)";
  pybind11::exec(command.c_str());
}

std::unique_ptr<edm::ParameterSet> PyBind11ProcessDesc::parameterSet() const {
  return std::make_unique<edm::ParameterSet>(theProcessPSet.pset());
}

std::string PyBind11ProcessDesc::dump() const {
  std::ostringstream os;
  os << theProcessPSet.dump();
  return os.str();
}

// For backward compatibility only.  Remove when no longer used.
std::unique_ptr<edm::ProcessDesc> PyBind11ProcessDesc::processDesc() const {
  return std::make_unique<edm::ProcessDesc>(parameterSet());
}
