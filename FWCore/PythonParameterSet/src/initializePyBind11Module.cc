// -*- C++ -*-
//
// Package:     PyBind11ParameterSet
// Class  :     initializePyBind11Module
//

#include "FWCore/PythonParameterSet/src/initializePyBind11Module.h"
#include "PyBind11Module.h"
#include <pybind11/embed.h>
#include <iostream>

namespace edm {
  namespace python {
    void initializePyBind11Module() {
      char *libFWCoreParameterSet = const_cast<char *>("libFWCorePythonParameterSet");
      pybind11::module::import(libFWCoreParameterSet);
    }
  }  // namespace python
}  // namespace edm
