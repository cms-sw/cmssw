#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/src/PythonModule.h"
#include "FWCore/ParameterSet/src/PythonWrapper.h"
#include <sstream>
#include <iostream>
#include <boost/foreach.hpp>
using namespace boost::python;

bool PythonProcessDesc::initialized_ = false;

PythonProcessDesc::PythonProcessDesc()
:  theProcessPSet(),
   theServices()
{
}


PythonProcessDesc::PythonProcessDesc(const std::string & config)
:  theProcessPSet(),
   theServices()
{
  PyImport_AppendInittab( "libFWCoreParameterSet", &initlibFWCoreParameterSet );
  Py_Initialize();
  if(!initialized_)
  {
    PyImport_ImportModule("libFWCoreParameterSet");
    initialized_ = true;
  }

  object main_module((
    handle<>(borrowed(PyImport_AddModule("__main__")))));

  object main_namespace = main_module.attr("__dict__");
  main_namespace["processDesc"] = ptr(this);
  main_namespace["processPSet"] = ptr(&theProcessPSet);

  try {
    // if it ends with py, it's a file
    if(config.substr(config.size()-3) == ".py")
    {
      readFile(config, main_namespace);
    }
    else
    {
      readString(config, main_namespace);
    }
  }
  catch( error_already_set ) {
     edm::pythonToCppException("Configuration");
     Py_Finalize();
  }
  Py_Finalize();
}


void PythonProcessDesc::readFile(const std::string & fileName, object & main_namespace)
{
  std::string initCommand("import FWCore.ParameterSet.Config as cms\n"
                      "execfile('");
  initCommand += fileName + "')";


  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
  std::string command("process.fillProcessDesc(processDesc, processPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
}


void PythonProcessDesc::readString(const std::string & pyConfig, object & main_namespace)
{
  std::string command = pyConfig;
  command += "\nprocess.fillProcessDesc(processDesc, processPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
}


boost::shared_ptr<edm::ProcessDesc> PythonProcessDesc::processDesc() const
{
  boost::shared_ptr<edm::ProcessDesc> result(new edm::ProcessDesc(theProcessPSet.pset()));
  BOOST_FOREACH(PythonParameterSet service, theServices)
  {
    result->addService(service.pset());
  }
  return result;
}
 

std::string PythonProcessDesc::dump() const
{
  std::ostringstream os;
  os << theProcessPSet.dump();
  BOOST_FOREACH(PythonParameterSet service, theServices)
  {
    os << service.dump();
  }
  return os.str();
}


