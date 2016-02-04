//
// Smart reference + singleton to handle Python interpreter
//
//
// Original Author:  Chris D Jones & Benedikt Hegner
//         Created:  Sun Jul 22 11:03:53 CEST 2006
//

#ifndef FWCore_Python_PythonManager_h
#define FWCore_Python_PythonManager_h

#include "FWCore/PythonParameterSet/interface/BoostPython.h"

extern "C" {
   //this is the entry point into the libFWCorePython python module
   void initlibFWCorePython();
   //void initROOT();
}

void pythonToCppException(const std::string& iType);

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
#endif // FWCore_Python_PythonManager_h
