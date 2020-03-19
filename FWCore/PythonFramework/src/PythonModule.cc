#ifndef FWCore_PythonFramework_PythonModule_h
#define FWCore_PythonFramework_PythonModule_h

#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"

#include "FWCore/PythonFramework/interface/PythonEventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <pybind11/pybind11.h>

// This is to give some special handling to cms::Exceptions thrown
// in C++ code called by python. Only at the very top level do
// we need the exception message returned by the function "what".
// We only need the central message here as this will get converted
// back into a cms::Exception again when control rises back into
// the C++ code.  If necessary it would probably be possible to
// improve these messages even more by adding something in the python
// to add module type and label context to the messages being caught
// here. At this point we did not think it worth the time to implement.

PYBIND11_MODULE(libFWCorePythonFramework, m) {
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const cms::Exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  pybind11::class_<PythonEventProcessor>(m, "PythonEventProcessor")
      .def(pybind11::init<PyBind11ProcessDesc const &>())
      .def("run", &PythonEventProcessor::run)
      .def("totalEvents", &PythonEventProcessor::totalEvents)
      .def("totalEventsPassed", &PythonEventProcessor::totalEventsPassed)
      .def("totalEventsFailed", &PythonEventProcessor::totalEventsFailed);
}
#endif
