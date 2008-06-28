#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/src/PythonModule.h"
#include "FWCore/ParameterSet/src/PythonWrapper.h"
#include <sstream>
using namespace boost::python;

PythonProcessDesc::PythonProcessDesc()
:  theProcessPSet(),
   theServices()
{
}


PythonProcessDesc::PythonProcessDesc(const std::string & config)
:  theProcessPSet(),
   theServices()
{
  // If given an argument, that argument must be the name of a file to read.
  Py_Initialize();
  PyImport_AppendInittab( "libFWCoreParameterSet", &initlibFWCoreParameterSet );


  object main_module((
    handle<>(borrowed(PyImport_AddModule("__main__")))));

  object main_namespace = main_module.attr("__dict__");

  // load the library
  object libModule( (handle<>(PyImport_ImportModule("libFWCoreParameterSet"))) );
  // put it in the main namespace
  main_namespace["libFWCoreParameterSet"] = libModule;

  // make an instance in python-land
  scope(libModule).attr("processDesc") = ptr(this);
  scope(libModule).attr("processPSet") = ptr(&theProcessPSet);
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
                      "fileDict = dict()\n"
                      "execfile('");
  initCommand += fileName + "',fileDict)";


  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
  std::string command("cms.findProcess(fileDict).fillProcessDesc(libFWCoreParameterSet.processDesc, libFWCoreParameterSet.processPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
}


void PythonProcessDesc::readString(const std::string & pyConfig, object & main_namespace)
{
  std::string command = pyConfig;
  command += "\nprocess.fillProcessDesc(libFWCoreParameterSet.processDesc, libFWCoreParameterSet.processPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        main_namespace.ptr(),
                        main_namespace.ptr()));
}


boost::shared_ptr<edm::ProcessDesc> PythonProcessDesc::processDesc() const
{
  boost::shared_ptr<edm::ProcessDesc> result(new edm::ProcessDesc(theProcessPSet.pset()));
  for(std::vector<PythonParameterSet>::const_iterator serviceItr = theServices.begin();
      serviceItr != theServices.end(); ++serviceItr)
  {
    result->addService(serviceItr->pset());
  }
  return result;
}
 

std::string PythonProcessDesc::dump() const
{
  std::ostringstream os;
  os << theProcessPSet.dump();
  for(std::vector<PythonParameterSet>::const_iterator serviceItr = theServices.begin();
      serviceItr != theServices.end(); ++serviceItr)
  {
    os << serviceItr->dump();
  }
  return os.str();
}


