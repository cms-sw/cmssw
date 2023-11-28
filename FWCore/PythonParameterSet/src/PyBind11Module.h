#ifndef FWCore_PythonParameterSet_PyBind11Module_h
#define FWCore_PythonParameterSet_PyBind11Module_h

#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <pybind11/stl.h>
#include <string>

// This is to give some special handling to cms::Exceptions thrown
// in C++ code called by python. Only at the very top level do
// we need the exception message returned by the function "what".
// We only need the central message here as this will get converted
// back into a cms::Exception again when control rises back into
// the C++ code.  If necessary it would probably be possible to
// improve these messages even more by adding something in the python
// to add module type and label context to the messages being caught
// here. At this point we did not think it worth the time to implement.
//namespace {
//  void translatorlibFWCorePythonParameterSet(cms::Exception const& ex) {
//    PyErr_SetString(PyExc_RuntimeError, ex.message().c_str());
//  }
//}

#include <pybind11/pybind11.h>

PYBIND11_MODULE(libFWCorePythonParameterSet, m) {
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const cms::Exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  pybind11::class_<edm::InputTag>(m, "InputTag")
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string &>())
      .def(pybind11::init<const std::string &, const std::string &, const std::string &>())
      .def(pybind11::init<const std::string &, const std::string &>())
      .def("label", &edm::InputTag::label, pybind11::return_value_policy::copy)
      .def("instance", &edm::InputTag::instance, pybind11::return_value_policy::copy)
      .def("process", &edm::InputTag::process, pybind11::return_value_policy::copy);

  pybind11::class_<edm::ESInputTag>(m, "ESInputTag")
      .def(pybind11::init<std::string>())
      .def(pybind11::init<std::string, std::string>())
      .def("module", &edm::ESInputTag::module, pybind11::return_value_policy::copy)
      .def("data", &edm::ESInputTag::data, pybind11::return_value_policy::copy);

  pybind11::class_<edm::EventID>(m, "EventID")
      .def(pybind11::init<edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t>())
      .def("run", &edm::EventID::run)
      .def("luminosityBlock", &edm::EventID::luminosityBlock)
      .def("event", &edm::EventID::event);

  pybind11::class_<edm::LuminosityBlockID>(m, "LuminosityBlockID")
      .def(pybind11::init<unsigned int, unsigned int>())
      .def("run", &edm::LuminosityBlockID::run)
      .def("luminosityBlock", &edm::LuminosityBlockID::luminosityBlock);

  pybind11::class_<edm::FileInPath>(m, "FileInPath")
      .def(pybind11::init<std::string>())
      .def("fullPath", &edm::FileInPath::fullPath)
      .def("relativePath", &edm::FileInPath::relativePath);

  pybind11::class_<edm::LuminosityBlockRange>(m, "LuminosityBlockRange")
      .def(pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def("start", &edm::LuminosityBlockRange::startRun)
      .def("startSub", &edm::LuminosityBlockRange::startLumi)
      .def("end", &edm::LuminosityBlockRange::endRun)
      .def("endSub", &edm::LuminosityBlockRange::endLumi);

  pybind11::class_<edm::EventRange>(m, "EventRange")
      .def(pybind11::init<edm::RunNumber_t,
                          edm::LuminosityBlockNumber_t,
                          edm::EventNumber_t,
                          edm::RunNumber_t,
                          edm::LuminosityBlockNumber_t,
                          edm::EventNumber_t>())
      .def("start", &edm::EventRange::startRun)
      .def("startLumi", &edm::EventRange::startLumi)
      .def("startSub", &edm::EventRange::startEvent)
      .def("end", &edm::EventRange::endRun)
      .def("endLumi", &edm::EventRange::endLumi)
      .def("endSub", &edm::EventRange::endEvent);

  pybind11::class_<Python11ParameterSet>(m, "ParameterSet")
      .def(pybind11::init<>())
      .def("addInt32", &Python11ParameterSet::addParameter<int>)
      .def("getInt32", &Python11ParameterSet::getParameter<int>)
      .def("addVInt32", &Python11ParameterSet::addParameters<int>)
      .def("getVInt32", &Python11ParameterSet::getParameters<int>)
      .def("addUInt32", &Python11ParameterSet::addParameter<unsigned int>)
      .def("getUInt32", &Python11ParameterSet::getParameter<unsigned int>)
      .def("addVUInt32", &Python11ParameterSet::addParameters<unsigned int>)
      .def("getVUInt32", &Python11ParameterSet::getParameters<unsigned int>)
      .def("addInt64", &Python11ParameterSet::addParameter<long long>)
      .def("getInt64", &Python11ParameterSet::getParameter<long long>)
      .def("addUInt64", &Python11ParameterSet::addParameter<unsigned long long>)
      .def("getUInt64", &Python11ParameterSet::getParameter<unsigned long long>)
      .def("addVInt64", &Python11ParameterSet::addParameters<long long>)
      .def("getVInt64", &Python11ParameterSet::getParameters<long long>)
      .def("addVUInt64", &Python11ParameterSet::addParameters<unsigned long long>)
      .def("getVUInt64", &Python11ParameterSet::getParameters<unsigned long long>)
      .def("addDouble", &Python11ParameterSet::addParameter<double>)
      .def("getDouble", &Python11ParameterSet::getParameter<double>)
      .def("addVDouble", &Python11ParameterSet::addParameters<double>)
      .def("getVDouble", &Python11ParameterSet::getParameters<double>)
      .def("addBool", &Python11ParameterSet::addParameter<bool>)
      .def("getBool", &Python11ParameterSet::getParameter<bool>)
      .def("addString", &Python11ParameterSet::addParameter<std::string>)
      .def("getString", &Python11ParameterSet::getParameter<std::string>)
      .def("addVString", &Python11ParameterSet::addParameters<std::string>)
      .def("getVString", &Python11ParameterSet::getParameters<std::string>)
      .def("addInputTag", &Python11ParameterSet::addParameter<edm::InputTag>)
      .def("getInputTag", &Python11ParameterSet::getParameter<edm::InputTag>)
      .def("addVInputTag", &Python11ParameterSet::addParameters<edm::InputTag>)
      .def("getVInputTag", &Python11ParameterSet::getParameters<edm::InputTag>)
      .def("addESInputTag", &Python11ParameterSet::addParameter<edm::ESInputTag>)
      .def("getESInputTag", &Python11ParameterSet::getParameter<edm::ESInputTag>)
      .def("addVESInputTag", &Python11ParameterSet::addParameters<edm::ESInputTag>)
      .def("getVESInputTag", &Python11ParameterSet::getParameters<edm::ESInputTag>)
      .def("addEventID", &Python11ParameterSet::addParameter<edm::EventID>)
      .def("getEventID", &Python11ParameterSet::getParameter<edm::EventID>)
      .def("addVEventID", &Python11ParameterSet::addParameters<edm::EventID>)
      .def("getVEventID", &Python11ParameterSet::getParameters<edm::EventID>)
      .def("addLuminosityBlockID", &Python11ParameterSet::addParameter<edm::LuminosityBlockID>)
      .def("getLuminosityBlockID", &Python11ParameterSet::getParameter<edm::LuminosityBlockID>)
      .def("addVLuminosityBlockID", &Python11ParameterSet::addParameters<edm::LuminosityBlockID>)
      .def("getVLuminosityBlockID", &Python11ParameterSet::getParameters<edm::LuminosityBlockID>)
      .def("addLuminosityBlockRange", &Python11ParameterSet::addParameter<edm::LuminosityBlockRange>)
      .def("getLuminosityBlockRange", &Python11ParameterSet::getParameter<edm::LuminosityBlockRange>)
      .def("addVLuminosityBlockRange", &Python11ParameterSet::addParameters<edm::LuminosityBlockRange>)
      .def("getVLuminosityBlockRange", &Python11ParameterSet::getParameters<edm::LuminosityBlockRange>)
      .def("addEventRange", &Python11ParameterSet::addParameter<edm::EventRange>)
      .def("getEventRange", &Python11ParameterSet::getParameter<edm::EventRange>)
      .def("addVEventRange", &Python11ParameterSet::addParameters<edm::EventRange>)
      .def("getVEventRange", &Python11ParameterSet::getParameters<edm::EventRange>)
      .def("addPSet", &Python11ParameterSet::addPSet)
      .def("getPSet", &Python11ParameterSet::getPSet)
      .def("addVPSet", &Python11ParameterSet::addVPSet)
      .def("getVPSet", &Python11ParameterSet::getVPSet)
      .def("addFileInPath", &Python11ParameterSet::addParameter<edm::FileInPath>)
      .def("getFileInPath", &Python11ParameterSet::getParameter<edm::FileInPath>)
      .def("newInputTag", &Python11ParameterSet::newInputTag)
      .def("newESInputTag", &Python11ParameterSet::newESInputTag)
      .def("newEventID", &Python11ParameterSet::newEventID)
      .def("newLuminosityBlockID", &Python11ParameterSet::newLuminosityBlockID)
      .def("newLuminosityBlockRange", &Python11ParameterSet::newLuminosityBlockRange)
      .def("newEventRange", &Python11ParameterSet::newEventRange)
      .def("addNewFileInPath", &Python11ParameterSet::addNewFileInPath)
      .def("newPSet", &Python11ParameterSet::newPSet)
      .def("dump", &Python11ParameterSet::dump);

  pybind11::class_<PyBind11ProcessDesc>(m, "ProcessDesc")  //, pybind11::init<>())
      .def(pybind11::init<>())
      .def(pybind11::init<std::string, bool>())
      .def("newPSet", &PyBind11ProcessDesc::newPSet)
      .def("pset", &PyBind11ProcessDesc::pset, pybind11::return_value_policy::reference)
      .def("dump", &PyBind11ProcessDesc::dump);
}

#endif
