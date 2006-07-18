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
//         Created:  Thu Mar 23 21:53:03 EST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/python.hpp"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Python/src/EventWrapper.h"
//
// class decleration
//
namespace {
   class PythonManagerHandle;
   class PythonManager {
      public:
	 friend class PythonManagerHandle;
	 static PythonManagerHandle handle();

      private:
	 PythonManager() : refCount_(0) {
	    //deactivate use of signal handling
	    Py_InitializeEx(0);
	 }
	 ~PythonManager() {
	    Py_Finalize();
	 }
	 void increment() {
	    ++refCount_;
	 }

	 void decrement() {
	    --refCount_;
	    if(0==refCount_) {
	       delete this;
	    }
	 }
	 unsigned long refCount_;
   };
   class PythonManagerHandle {
      public:
	 ~PythonManagerHandle() { manager_.decrement(); }

	 PythonManagerHandle(PythonManager& iM):
	    manager_(iM) {
	    manager_.increment();
	 }

	 PythonManagerHandle(const PythonManagerHandle& iRHS) :
	    manager_(iRHS.manager_) {
	    manager_.increment();
	 }
      private:
	 const PythonManagerHandle& operator=(const PythonManagerHandle&);

	 PythonManager& manager_;
   };

   PythonManagerHandle PythonManager::handle() {
      static PythonManager* s_manager( new PythonManager() );
      return PythonManagerHandle( *s_manager);
   }
}


class PythonFilter : public edm::EDFilter {
   public:
      explicit PythonFilter(const edm::ParameterSet&);
      ~PythonFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      std::string command_; 
      //have this first to guarantee that Py_Finalize not called
      // until the object's destructor is called
      PythonManagerHandle handle_;
      boost::python::object filter_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
extern "C" {
   //this is the entry point into the libFWCorePython python module
   void initlibFWCorePython();
   //void initROOT();
}

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
//
// constructors and destructor
//

//NOTE: need to add ROOTSYS/lib to PYTHONPATH

PythonFilter::PythonFilter(const edm::ParameterSet& iConfig) :
   command_("import sys\n"
	    "sys.argv=['']\n"  //ROOT module uses this so must be set
	    "import os\n"
	    "if os.environ.has_key('ROOTSYS'):\n" //ROOT module is in $ROOTSYS/lib
	    "  sys.path.append(os.environ['ROOTSYS']+'/lib')\n"
	    "import ROOT\n"
	    "ROOT.gSystem.Load(\"libPhysicsToolsFWLite\")\n"
	    "ROOT.AutoLibraryLoader.enable()\n"
	    "import libFWCorePython as edm\n"),
   handle_(PythonManager::handle())
{
   const std::vector<std::string> commandLines = iConfig.getParameter<std::vector<std::string> >("command");
   
   for( std::vector<std::string>::const_iterator itLine = commandLines.begin();
	itLine != commandLines.end();
	++itLine) {
      command_ += *itLine;
      command_ += "\n";
   }
   //now do what ever initialization is needed
   //Py_Initialize();

   using namespace boost::python;
   //make sure our custom module gets loaded
   if(PyImport_AppendInittab("libFWCorePython",initlibFWCorePython)==-1) {
      throw cms::Exception("InitializationFailure" )
	 <<"failed to add libFWCorePython python module to python interpreter";
   }

   object main_module((
			 handle<>(borrowed(PyImport_AddModule("__main__")))));
   
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

   } catch( error_already_set ) {
      pythonToCppException("Configuration");
   }
}


PythonFilter::~PythonFilter()
{
 
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
PythonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace boost::python;
   object main_module((
			 handle<>(borrowed(PyImport_AddModule("__main__")))));
   
   object main_namespace = main_module.attr("__dict__");

   main_namespace["event"] = object(edm::python::ConstEventWrapper(iEvent) );
   main_namespace["tempFilter"] = filter_;

   try {
      object result((handle<>(PyRun_String("tempFilter.filter(event)",
					   Py_eval_input,
					   main_namespace.ptr(),
					   main_namespace.ptr()))));
      return extract<bool>(result);
      
   }catch( error_already_set ) {
      pythonToCppException("RuntimeError");
   }
   return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(PythonFilter)
