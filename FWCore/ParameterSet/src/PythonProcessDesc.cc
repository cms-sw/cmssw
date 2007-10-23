#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/src/PythonModule.h"
#include "FWCore/ParameterSet/src/PythonWrapper.h"
#include <boost/python.hpp>
#include <sstream>

#include <iostream>
PythonProcessDesc::PythonProcessDesc()
:  theProcessPSet(),
   theServices()
{
}


PythonProcessDesc::PythonProcessDesc(const std::string & fileName)
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
  catch( error_already_set ) {
     edm::pythonToCppException("Configuration");
     Py_Finalize();
  }

  Py_Finalize();
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


