#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PyDevParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/PyDevParameterSet/src/initializePyBind11Module.h"
#include "FWCore/PyDevParameterSet/src/PyBind11Wrapper.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <iostream>

PyBind11ProcessDesc::PyBind11ProcessDesc() :
   theProcessPSet(),
   theMainModule()
{
  pybind11::initialize_interpreter();
}

PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config) :
   theProcessPSet(),
   theMainModule()
{
  pybind11::initialize_interpreter();
  edm::python::initializePyBind11Module();
  prepareToRead();
  read(config);
}

/*
PyBind11ProcessDesc::PyBind11ProcessDesc(std::string const& config, int argc, char* argv[]) :
   theProcessPSet(),
   theMainModule()
{
  pybind11::initialize_interpreter();
  edm::python::initializePyBind11Module();
  prepareToRead();
  PySys_SetArgv(argc, argv);
  read(config);
  //py::module::import("sys").attr("argv") = py::make_tuple("test.py", "embed.cpp"); might work
}
*/

PyBind11ProcessDesc::~PyBind11ProcessDesc() {
  theMainModule=pybind11::object();
  pybind11::finalize_interpreter();

}

void PyBind11ProcessDesc::prepareToRead() {
  //  pybind11::scoped_interpreter guard{};
  theMainModule = pybind11::module::import("__main__");
  theMainModule.attr("processDesc")=this; 
  theMainModule.attr("processPSet")=&theProcessPSet; 
}

void PyBind11ProcessDesc::read(std::string const& config) {
  try {
    // if it ends with py, it's a file
    if(config.substr(config.size()-3) == ".py") {
      readFile(config);
    } else {
      readString(config);
    }
  }
  catch(pybind11::error_already_set const&e) {
    edm::pythonToCppException("Configuration",e.what());
  }
}

void PyBind11ProcessDesc::readFile(std::string const& fileName) {
  std::string initCommand("import FWCore.ParameterSet.Config as cms\n"
                          "execfile('");
  initCommand += fileName + "')";

  pybind11::exec(initCommand.c_str());
  std::string command("process.fillProcessDesc(processPSet)");
  pybind11::exec(command.c_str());
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
