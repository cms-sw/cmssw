#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/PythonParameterSet/src/initializeModule.h"
#include "FWCore/PythonParameterSet/src/PythonWrapper.h"

#include <sstream>

using namespace boost::python;

PythonProcessDesc::PythonProcessDesc() :
   theProcessPSet(),
   theMainModule(),
   theMainNamespace() {
}

PythonProcessDesc::PythonProcessDesc(std::string const& config) :
   theProcessPSet(),
   theMainModule(),
   theMainNamespace() {
  prepareToRead();
  read(config);
  Py_Finalize();
}

PythonProcessDesc::PythonProcessDesc(std::string const& config, int argc, char* argv[]) :
   theProcessPSet(),
   theMainModule(),
   theMainNamespace() {
  prepareToRead();
  PySys_SetArgv(argc, argv);
  read(config);
  Py_Finalize();
}

void PythonProcessDesc::prepareToRead() {
  edm::python::initializeModule();

  theMainModule = object(handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__")))));

  theMainNamespace = theMainModule.attr("__dict__");
  theMainNamespace["processDesc"] = ptr(this);
  theMainNamespace["processPSet"] = ptr(&theProcessPSet);
}

void PythonProcessDesc::read(std::string const& config) {
  try {
    // if it ends with py, it's a file
    if(config.substr(config.size()-3) == ".py") {
      readFile(config);
    } else {
      readString(config);
    }
  }
  catch(error_already_set) {
     edm::pythonToCppException("Configuration");
     Py_Finalize();
  }
}

void PythonProcessDesc::readFile(std::string const& fileName) {
  std::string initCommand("import FWCore.ParameterSet.Config as cms\n"
                          "execfile('");
  initCommand += fileName + "')";

  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
  std::string command("process.fillProcessDesc(processPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
}

void PythonProcessDesc::readString(std::string const& pyConfig) {
  std::string command = pyConfig;
  command += "\nprocess.fillProcessDesc(processPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
}

boost::shared_ptr<edm::ParameterSet> PythonProcessDesc::parameterSet() {
  boost::shared_ptr<edm::ParameterSet> result(new edm::ParameterSet(theProcessPSet.pset()));
  return result;
}

std::string PythonProcessDesc::dump() const {
  std::ostringstream os;
  os << theProcessPSet.dump();
  return os.str();
}
