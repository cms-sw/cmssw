// -*- C++ -*-
//
// Package:     PyBind11ParameterSet
// Class  :     initializePyBind11Module
//

#include "FWCore/PythonParameterSet/src/initializePyBind11Module.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PyBind11Module.h"
#include <pybind11/embed.h>
#include <iostream>

namespace edm {
  namespace python {
    void initializePyBind11Module() {
      char *libFWCoreParameterSet = const_cast<char *>("libFWCorePythonParameterSet");
      try {
        pybind11::module::import(libFWCoreParameterSet);
      } catch (const std::exception &e) {
        throw cms::Exception("PyBind11ImportError") << "Failed to import PyBind11 module: " << e.what();
      }
    }
  }  // namespace python
}  // namespace edm
