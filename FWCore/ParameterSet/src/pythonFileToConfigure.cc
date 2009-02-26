/*
 *  pythonToConfigure.cpp
 *  CMSSW
 *
 *  Created by Chris Jones on 10/3/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

#include "FWCore/ParameterSet/interface/BoostPython.h"
#include "FWCore/ParameterSet/interface/pythonFileToConfigure.h"
#include "FWCore/ParameterSet/src/PythonWrapper.h"


std::string edm::pythonFileToConfigure(const std::string& iPythonFileName)
{
  std::string returnValue;
  std::string initCommand("import FWCore.ParameterSet.Config as cms\n"
                          "fileDict = dict()\n"
                          "execfile('");
  initCommand += iPythonFileName+"',fileDict)";
  
  Py_InitializeEx(0);
  using namespace boost::python;
  
  object main_module((
                      boost::python::handle<PyObject>(borrowed(PyImport_AddModule("__main__")))));
  object main_namespace = main_module.attr("__dict__");
  try {
    try {
      object result((boost::python::handle<>(PyRun_String(initCommand.c_str(),
                                                          Py_file_input,
                                                          main_namespace.ptr(),
                                                          main_namespace.ptr()))));
    } catch(error_already_set) {
      edm::pythonToCppException("Configuration");
    }
    try {
      std::string command("cms.findProcess(fileDict).dumpConfig()");
      object result((handle<>(PyRun_String(command.c_str(),
                                           Py_eval_input,
                                           main_namespace.ptr(),
                                           main_namespace.ptr()))));
      returnValue= extract<std::string>(result);
      
    }catch( error_already_set ) {
      edm::pythonToCppException("Configuration");
    }
  }catch(...) {
    Py_Finalize();
    throw;
  }
  Py_Finalize();
  return returnValue;
}

