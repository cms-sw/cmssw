// -*- C++ -*-
//
// Package:     PyBind11ParameterSet
// Class  :     initializePyBind11Module
// 

// system include files

// user include files
#include "FWCore/PyBind11ParameterSet/src/initializePyBind11Module.h"
#include "PyBind11Module.h"
#include <pybind11/embed.h>
#include <iostream>
//
// constants, enums and typedefs
//

namespace edm {
   namespace python {
      void initializePyBind11Module() {
         char *libFWCoreParameterSet = const_cast<char *>("libFWCorePyBind11ParameterSet");
         PyImport_AppendInittab(libFWCoreParameterSet, &initlibFWCorePyBind11ParameterSet );
	 PyImport_ImportModule(libFWCoreParameterSet);
      }
   }
}
