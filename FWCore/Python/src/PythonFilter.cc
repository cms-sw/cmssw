// -*- C++ -*-
//
// Package:    PythonFilter
// Class:      PythonFilter
// 
/**\class PythonFilter PythonFilter.cc FWCore/PythonFilter/src/PythonFilter.cc

 Description: an EDFilter which uses python code to do the work

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Thu Mar 23 21:53:03 CEST 2006
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

// subpackage specific includes
#include "FWCore/Python/src/EventWrapper.h"
#include "FWCore/Python/src/PythonFilter.h"

// system include files
#include <memory>

//
// constructors and destructor
//

PythonFilter::PythonFilter(edm::ParameterSet const& iConfig) :
 //  command_("import sys\n"
//	    "sys.argv=['']\n"  //ROOT module uses this so must be set
//	    "import ROOT\n"
//	    "ROOT.gSystem.Load(\"libFWCoreFWLite\")\n"
//	    "ROOT.AutoLibraryLoader.enable()\n"
//	    "import libFWCorePython as edm\n"),
   handle_(PythonManager::handle()) {
   std::vector<std::string> const commandLines = iConfig.getParameter<std::vector<std::string> >("command");
   
   for(std::vector<std::string>::const_iterator itLine = commandLines.begin();
	itLine != commandLines.end();
	++itLine) {
      command_ += *itLine;
      command_ += "\n";
   }

   using namespace boost::python;
   //make sure our custom module gets loaded
   //if(PyImport_AppendInittab("libFWCorePython",initlibFWCorePython)==-1) {
   //   throw cms::Exception("InitializationFailure")
   //<<"failed to add libFWCorePython python module to python interpreter";
   //}

   object main_module((handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__"))))));
   
   object main_namespace = main_module.attr("__dict__");

   try {
      object result((handle<>(PyRun_String(command_.c_str(),
					   Py_file_input,
					   main_namespace.ptr(),
					   main_namespace.ptr()))));
      try {
	 filter_ = main_namespace["filter"];
      } catch(...) {
	 throw cms::Exception("Configuration") <<"No 'filter' python variable defined in 'command' parameter.\n Please create an instance of the python class you want to use for filtering and pass that instance to the variable named 'filter'.";
      }

   } catch(error_already_set) {
      pythonToCppException("Configuration");
   }
}

PythonFilter::~PythonFilter() {
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   //filter_ = boost::python::object();
   //Py_Finalize();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
PythonFilter::filter(edm::Event& iEvent, edm::EventSetup const&) {
   using namespace boost::python;
   object main_module((
			 handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__"))))));
   
   object main_namespace = main_module.attr("__dict__");

   main_namespace["event"] = object(edm::python::ConstEventWrapper(iEvent));
   main_namespace["tempFilter"] = filter_;

   try {
      object result((handle<>(PyRun_String("tempFilter.filter(event)",
					   Py_eval_input,
					   main_namespace.ptr(),
					   main_namespace.ptr()))));
      return extract<bool>(result);
      
   } catch(error_already_set) {
      pythonToCppException("RuntimeError");
   }
   return false;
}

