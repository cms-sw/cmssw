//
// Smart reference + singleton to handle Python interpreter
//
//
// Original Author:  Chris D Jones & Benedikt Hegner
//         Created:  Sun Jul 22 11:03:53 EST 2006
//
// $Id: PythonManager.h,v 1.1 2006/07/22 13:17:19 hegner Exp $

#ifndef Python_Manager_h
#define Python_Manager_h

#include "FWCore/Utilities/interface/Exception.h"

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

#endif //Python_Manager_h
