// -*- C++ -*-
//
// Package:     PythonParameterSet
// Class  :     initializeModule
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 11 11:05:58 CST 2011
// $Id: initializeModule.cc,v 1.1 2011/01/11 19:25:55 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/PythonParameterSet/src/initializeModule.h"
#include "FWCore/PythonParameterSet/src/PythonModule.h"


//
// constants, enums and typedefs
//
static bool s_initialized = false;

namespace edm {
   namespace python {
      void initializeModule() {
         char *libFWCoreParameterSet = const_cast<char *>("libFWCoreParameterSet");
         PyImport_AppendInittab(libFWCoreParameterSet, &initlibFWCoreParameterSet );
         Py_Initialize();
         if(!s_initialized)
         {
            PyImport_ImportModule(libFWCoreParameterSet);
            s_initialized = true;
         }
         
      }
   }
}
