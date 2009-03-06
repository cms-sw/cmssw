//
// Smart reference + singleton to handle Python interpreter
//
//
// Original Author:  Chris D Jones & Benedikt Hegner
//         Created:  Sun Jul 22 11:03:53 CEST 2006
//

#ifndef FWCore_Python_PythonManager_h
#define FWCore_Python_PythonManager_h

#include "FWCore/ParameterSet/interface/BoostPython.h"
#include "FWCore/Utilities/interface/Exception.h"

extern "C" {
   //this is the entry point into the libFWCorePython python module
   void initlibFWCorePython();
   //void initROOT();
}

namespace {
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

    class PythonManagerHandle;
  	
    class PythonManager {
      public:
	    friend class PythonManagerHandle;
	    static PythonManagerHandle handle();

      private:
	     PythonManager();
	     ~PythonManager() { Py_Finalize(); }
	     void increment() { ++refCount_; }
         void decrement() { --refCount_; if(0==refCount_) delete this; }
	     unsigned long refCount_;
	     std::string initCommand_;
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

    PythonManager::PythonManager() :
        refCount_(0),
        initCommand_(
            "import sys\n"
	    "sys.path.append('./')\n"
            "import ROOT\n"
            "ROOT.gSystem.Load(\"libFWCoreFWLite\")\n"
            "ROOT.AutoLibraryLoader.enable()\n"
            "import libFWCorePython as edm\n")
    {
       Py_InitializeEx(0);
       using namespace boost::python;

       if(PyImport_AppendInittab("libFWCorePython",initlibFWCorePython)==-1) {
         throw cms::Exception("InitializationFailure" )
	  <<"failed to add libFWCorePython python module to python interpreter";
       }
       object main_module((
                           boost::python::handle<PyObject>(borrowed(PyImport_AddModule("__main__")))));
       object main_namespace = main_module.attr("__dict__");
       try {
           object result((boost::python::handle<>(PyRun_String(initCommand_.c_str(),
		    			   Py_file_input,
					   main_namespace.ptr(),
					   main_namespace.ptr()))));

       } catch(...  ) {
	 throw cms::Exception("Configuration") << "test";
       }

    }
}

#endif // FWCore_Python_PythonManager_h
