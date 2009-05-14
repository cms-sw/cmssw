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
   theServices(),
   theMainModule(),
   theMainNamespace()
{
}


PythonProcessDesc::PythonProcessDesc(std::string const& config)
:  theProcessPSet(),
   theServices(),
   theMainModule(),
   theMainNamespace()
{
  prepareToRead();
  read(config);
  Py_Finalize();
}

PythonProcessDesc::PythonProcessDesc(std::string const& config, int argc, char * argv[])
:  theProcessPSet(),
   theServices(),
   theMainModule(),
   theMainNamespace()
{
  prepareToRead();
  PySys_SetArgv(argc, argv);
  read(config);
  Py_Finalize();
}


void PythonProcessDesc::prepareToRead()
{
  PyImport_AppendInittab( "libFWCoreParameterSet", &initlibFWCoreParameterSet );
  Py_Initialize();
  if(!initialized_)
  {
    PyImport_ImportModule("libFWCoreParameterSet");
    initialized_ = true;
  }

  theMainModule = object(handle<>(borrowed(PyImport_AddModule("__main__"))));

  theMainNamespace = theMainModule.attr("__dict__");
  theMainNamespace["processDesc"] = ptr(this);
  theMainNamespace["processPSet"] = ptr(&theProcessPSet);
}


void PythonProcessDesc::read(std::string const& config)
{  
  try {
    // if it ends with py, it's a file
    if(config.substr(config.size()-3) == ".py")
    {
      readFile(config);
    }
    else
    {
      readString(config);
    }
  }
  catch( error_already_set ) {
     edm::pythonToCppException("Configuration");
     Py_Finalize();
  }
}


void PythonProcessDesc::readFile(std::string const& fileName)
{
  std::string initCommand("import FWCore.ParameterSet.Config as cms\n"
                      "execfile('");
  initCommand += fileName + "')";


  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
  std::string command("process.fillProcessDesc(processDesc, processPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
}


void PythonProcessDesc::readString(std::string const& pyConfig)
{
  std::string command = pyConfig;
  command += "\nprocess.fillProcessDesc(processDesc, processPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        theMainNamespace.ptr(),
                        theMainNamespace.ptr()));
}


boost::shared_ptr<edm::ProcessDesc> PythonProcessDesc::processDesc()
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


