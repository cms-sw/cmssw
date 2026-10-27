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

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config, bool isFile, ProcessOptions const& options)
    : theProcessPSet(), theInterpreter(true) {
  edm::python::initializePyBind11Module();
  prepareToRead();
  read(config, isFile, options);
}

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config,
                                         bool isFile,
                                         const std::vector<std::string>& args,
                                         ProcessOptions const& options)
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
  read(config, isFile, options);
}

PyBind11ProcessDesc::~PyBind11ProcessDesc() = default;

void PyBind11ProcessDesc::prepareToRead() {
  //  pybind11::scoped_interpreter guard{};
  theInterpreter.mainModule = pybind11::module::import("__main__");
  theInterpreter.mainModule.attr("processDesc") = this;
  theInterpreter.mainModule.attr("processPSet") = &theProcessPSet;
}

void PyBind11ProcessDesc::read(std::string const& config, bool isFile, ProcessOptions const& options) {
  try {
    if (isFile)
      pybind11::eval_file(config);
    else
      pybind11::exec(config);
    setProcessOptions(options);
    pybind11::exec("process.fillProcessDesc(processPSet)");
  } catch (pybind11::error_already_set const& e) {
    edm::pythonToCppException("Configuration", e.what());
  }
}

void PyBind11ProcessDesc::setProcessOptions(ProcessOptions const& options) {
  if (options.empty())
    return;

  pybind11::dict globals = pybind11::globals();
  if (not globals.contains("process")) {
    throw cms::Exception("Configuration")
        << "Cannot set the process options, the configuration does not define a 'process' object.";
  }
  pybind11::object processOptions = globals["process"].attr("options");
  pybind11::object validNames = processOptions.attr("parameterNames_")();

  for (auto const& [name, value] : options) {
    // split nested options like "eventSetup.numberOfConcurrentIOVs" into their components
    std::vector<std::string> path;
    std::string::size_type begin = 0;
    std::string::size_type end;
    do {
      end = name.find('.', begin);
      path.emplace_back(name.substr(begin, end - begin));
      begin = end + 1;
    } while (end != std::string::npos);

    if (not validNames.contains(path.front())) {
      throw cms::Exception("Configuration") << "Invalid process option \"" << name << " = " << value << "\":\n\""
                                            << path.front() << "\" is not a valid process option.";
    }

    try {
      pybind11::object target = processOptions;
      for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        target = target.attr(path[i].c_str());
      }
      pybind11::object pyValue = pybind11::eval(value, globals);
      // for convenience, a single value can be assigned to a vector option, e.g. accelerators = "gpu-*"
      char const* attr = path.back().c_str();
      if (pybind11::hasattr(target, attr) and pybind11::isinstance<pybind11::list>(target.attr(attr)) and
          not pybind11::isinstance<pybind11::list>(pyValue) and not pybind11::isinstance<pybind11::tuple>(pyValue)) {
        pybind11::list values;
        values.append(pyValue);
        pyValue = values;
      }
      pybind11::setattr(target, attr, pyValue);
    } catch (pybind11::error_already_set const& e) {
      throw cms::Exception("Configuration") << "Invalid process option \"" << name << " = " << value << "\":\n"
                                            << e.what();
    }
  }
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
