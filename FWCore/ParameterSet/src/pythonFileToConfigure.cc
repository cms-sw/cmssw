/*
 *  pythonToConfigure.cpp
 *  CMSSW
 *
 *  Created by Chris Jones on 10/3/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

#include <boost/python.hpp>
#include "FWCore/ParameterSet/interface/pythonFileToConfigure.h"
#include "FWCore/Utilities/interface/Exception.h"


static
void
pythonToCppException(const std::string& iType)
{
  using namespace boost::python;
  PyObject *exc, *val, *trace;
  PyErr_Fetch(&exc,&val,&trace);
  handle<> hExc(allow_null(exc));
  if(hExc) {
    object oExc(hExc);
  }
  handle<> hVal(allow_null(val));
  handle<> hTrace(allow_null(trace));
  if(hTrace) {
    object oTrace(hTrace);
  }
  
  if(hVal) {
    object oVal(hVal);
    handle<> hStringVal(PyObject_Str(oVal.ptr()));
    object stringVal( hStringVal );
    
    //PyErr_Print();
    throw cms::Exception(iType) <<"python encountered the error: "<< PyString_AsString(stringVal.ptr())<<"\n";
  } else {
    throw cms::Exception(iType)<<" unknown python problem occurred.\n";
  }
}


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
      pythonToCppException("Configuration");
    }
    try {
      std::string command("cms.findProcess(fileDict).dumpConfig()");
      object result((handle<>(PyRun_String(command.c_str(),
                                           Py_eval_input,
                                           main_namespace.ptr(),
                                           main_namespace.ptr()))));
      returnValue= extract<std::string>(result);
      
    }catch( error_already_set ) {
      pythonToCppException("Configuration");
    }
  }catch(...) {
    Py_Finalize();
    throw;
  }
  Py_Finalize();
  return returnValue;
}

