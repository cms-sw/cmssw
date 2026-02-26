#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/PythonParameterSet/src/initializePyBind11Module.h"
#include "FWCore/PythonParameterSet/interface/PyBind11Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <filesystem>
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

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config, bool isFile)
    : theProcessPSet(), theInterpreter(true) {
  edm::python::initializePyBind11Module();
  prepareToRead();
  read(config, isFile);
}

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config, bool isFile, const std::vector<std::string>& args)
    : theProcessPSet(), theInterpreter(true) {
  edm::python::initializePyBind11Module();
  prepareToRead();

  // Acquire Python GIL before touching Python objects
  pybind11::gil_scoped_acquire acquire;

  try {
    // Import sys module
    pybind11::module_ sys = pybind11::module_::import("sys");

    // set sys.argv safely
    pybind11::list pyargs;
    for (auto const& arg : args) {
      pyargs.append(arg);
    }
    sys.attr("argv") = pyargs;

    // emulate old implicit sys.path[0] update (PySys_SetArgv behaviour)
    if (!args.empty()) {
      std::string dir = std::filesystem::path(args[0]).parent_path().string();
      if (dir.empty())
        dir = ".";  // fallback to current directory

      // insert dir at the front of sys.path
      pybind11::list sys_path = sys.attr("path");
      sys_path.insert(0, dir);
      sys.attr("path") = sys_path;
    }
  } catch (const pybind11::error_already_set& e) {
    throw cms::Exception("PyBind11Error") << "Python error initializing PyBind11ProcessDesc: " << e.what();
  }
  read(config, isFile);
}

PyBind11ProcessDesc::~PyBind11ProcessDesc() = default;

void PyBind11ProcessDesc::prepareToRead() {
  //  pybind11::scoped_interpreter guard{};
  theInterpreter.mainModule = pybind11::module::import("__main__");
  theInterpreter.mainModule.attr("processDesc") = this;
  theInterpreter.mainModule.attr("processPSet") = &theProcessPSet;
}

void PyBind11ProcessDesc::read(std::string const& config, bool isFile) {
  try {
    if (isFile)
      readFile(config);
    else
      readString(config);
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
