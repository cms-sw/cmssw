#ifndef FWCore_PythonParameterSet_PythonModule_h
#define FWCore_PythonParameterSet_PythonModule_h

#include "FWCore/PythonParameterSet/interface/BoostPython.h"
#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
namespace {
  void translator(cms::Exception const& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.message().c_str());
  }
}

BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{
  boost::python::register_exception_translator<cms::Exception>(translator);

  boost::python::class_<edm::InputTag>("InputTag", boost::python::init<std::string>())
      .def(boost::python::init<std::string, std::string, std::string>())
      .def(boost::python::init<std::string, std::string>())
      .def("label",    &edm::InputTag::label, boost::python::return_value_policy<boost::python::copy_const_reference>())
      .def("instance", &edm::InputTag::instance, boost::python::return_value_policy<boost::python::copy_const_reference>())
      .def("process",  &edm::InputTag::process, boost::python::return_value_policy<boost::python::copy_const_reference>())
  ;

   boost::python::class_<edm::ESInputTag>("ESInputTag", boost::python::init<std::string>())
   .def(boost::python::init<std::string, std::string>())
   .def("module",    &edm::ESInputTag::module, boost::python::return_value_policy<boost::python::copy_const_reference>())
   .def("data",  &edm::ESInputTag::data, boost::python::return_value_policy<boost::python::copy_const_reference>())
   ;

   boost::python::class_<edm::EventID>("EventID", boost::python::init<unsigned int, unsigned int, unsigned int>())
      .def("run",   &edm::EventID::run)
      .def("luminosityBlock", &edm::EventID::luminosityBlock)
      .def("event", &edm::EventID::event)
  ;

  boost::python::class_<edm::LuminosityBlockID>("LuminosityBlockID", boost::python::init<unsigned int, unsigned int>())
      .def("run",    &edm::LuminosityBlockID::run)
      .def("luminosityBlock", &edm::LuminosityBlockID::luminosityBlock)
  ;

  boost::python::class_<edm::FileInPath>("FileInPath", boost::python::init<std::string>())
      .def("fullPath",     &edm::FileInPath::fullPath)
      .def("relativePath", &edm::FileInPath::relativePath)
  ;

  boost::python::class_<edm::LuminosityBlockRange>("LuminosityBlockRange", boost::python::init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def("start",    &edm::LuminosityBlockRange::startRun)
      .def("startSub", &edm::LuminosityBlockRange::startLumi)
      .def("end",      &edm::LuminosityBlockRange::endRun)
      .def("endSub",   &edm::LuminosityBlockRange::endLumi)
  ;

  boost::python::class_<edm::EventRange>("EventRange", boost::python::init<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>())
      .def("start",     &edm::EventRange::startRun)
      .def("startLumi", &edm::EventRange::startLumi)
      .def("startSub",  &edm::EventRange::startEvent)
      .def("end",       &edm::EventRange::endRun)
      .def("endLumi",   &edm::EventRange::endLumi)
      .def("endSub",    &edm::EventRange::endEvent)
  ;

  boost::python::class_<PythonParameterSet>("ParameterSet")
    .def("addInt32", &PythonParameterSet::addParameter<int>)
    .def("getInt32", &PythonParameterSet::getParameter<int>)
    .def("addVInt32", &PythonParameterSet::addParameters<int>)
    .def("getVInt32", &PythonParameterSet::getParameters<int>)
    .def("addUInt32", &PythonParameterSet::addParameter<unsigned int>)
    .def("getUInt32", &PythonParameterSet::getParameter<unsigned int>)
    .def("addVUInt32", &PythonParameterSet::addParameters<unsigned int>)
    .def("getVUInt32", &PythonParameterSet::getParameters<unsigned int>)
    .def("addInt64", &PythonParameterSet::addParameter<long long>)
    .def("getInt64", &PythonParameterSet::getParameter<long long>)
    .def("addUInt64", &PythonParameterSet::addParameter<unsigned long long>)
    .def("getUInt64", &PythonParameterSet::getParameter<unsigned long long>)
    .def("addVInt64", &PythonParameterSet::addParameters<long long>)
    .def("getVInt64", &PythonParameterSet::getParameters<long long>)
    .def("addVUInt64", &PythonParameterSet::addParameters<unsigned long long>)
    .def("getVUInt64", &PythonParameterSet::getParameters<unsigned long long>)
    .def("addDouble", &PythonParameterSet::addParameter<double>)
    .def("getDouble", &PythonParameterSet::getParameter<double>)
    .def("addVDouble", &PythonParameterSet::addParameters<double>)
    .def("getVDouble", &PythonParameterSet::getParameters<double>)
    .def("addBool", &PythonParameterSet::addParameter<bool>)
    .def("getBool", &PythonParameterSet::getParameter<bool>)
    .def("addString", &PythonParameterSet::addParameter<std::string>)
    .def("getString", &PythonParameterSet::getParameter<std::string>)
    .def("addVString", &PythonParameterSet::addParameters<std::string>)
    .def("getVString", &PythonParameterSet::getParameters<std::string>)
    .def("addInputTag", &PythonParameterSet::addParameter<edm::InputTag>)
    .def("getInputTag", &PythonParameterSet::getParameter<edm::InputTag>)
    .def("addVInputTag", &PythonParameterSet::addParameters<edm::InputTag>)
    .def("getVInputTag", &PythonParameterSet::getParameters<edm::InputTag>)
    .def("addESInputTag", &PythonParameterSet::addParameter<edm::ESInputTag>)
    .def("getESInputTag", &PythonParameterSet::getParameter<edm::ESInputTag>)
    .def("addVESInputTag", &PythonParameterSet::addParameters<edm::ESInputTag>)
    .def("getVESInputTag", &PythonParameterSet::getParameters<edm::ESInputTag>)
    .def("addEventID", &PythonParameterSet::addParameter<edm::EventID>)
    .def("getEventID", &PythonParameterSet::getParameter<edm::EventID>)
    .def("addVEventID", &PythonParameterSet::addParameters<edm::EventID>)
    .def("getVEventID", &PythonParameterSet::getParameters<edm::EventID>)
    .def("addLuminosityBlockID", &PythonParameterSet::addParameter<edm::LuminosityBlockID>)
    .def("getLuminosityBlockID", &PythonParameterSet::getParameter<edm::LuminosityBlockID>)
    .def("addVLuminosityBlockID", &PythonParameterSet::addParameters<edm::LuminosityBlockID>)
    .def("getVLuminosityBlockID", &PythonParameterSet::getParameters<edm::LuminosityBlockID>)
    .def("addLuminosityBlockRange", &PythonParameterSet::addParameter<edm::LuminosityBlockRange>)
    .def("getLuminosityBlockRange", &PythonParameterSet::getParameter<edm::LuminosityBlockRange>)
    .def("addVLuminosityBlockRange", &PythonParameterSet::addParameters<edm::LuminosityBlockRange>)
    .def("getVLuminosityBlockRange", &PythonParameterSet::getParameters<edm::LuminosityBlockRange>)
    .def("addEventRange", &PythonParameterSet::addParameter<edm::EventRange>)
    .def("getEventRange", &PythonParameterSet::getParameter<edm::EventRange>)
    .def("addVEventRange", &PythonParameterSet::addParameters<edm::EventRange>)
    .def("getVEventRange", &PythonParameterSet::getParameters<edm::EventRange>)
    .def("addPSet", &PythonParameterSet::addPSet)
    .def("getPSet", &PythonParameterSet::getPSet)
    .def("addVPSet", &PythonParameterSet::addVPSet)
    .def("getVPSet", &PythonParameterSet::getVPSet)
    .def("addFileInPath", &PythonParameterSet::addParameter<edm::FileInPath>)
    .def("getFileInPath", &PythonParameterSet::getParameter<edm::FileInPath>)
    .def("newInputTag", &PythonParameterSet::newInputTag)
    .def("newESInputTag", &PythonParameterSet::newESInputTag)
    .def("newEventID", &PythonParameterSet::newEventID)
    .def("newLuminosityBlockID", &PythonParameterSet::newLuminosityBlockID)
    .def("newLuminosityBlockRange", &PythonParameterSet::newLuminosityBlockRange)
    .def("newEventRange", &PythonParameterSet::newEventRange)
    .def("addNewFileInPath", &PythonParameterSet::addNewFileInPath)
    .def("newPSet", &PythonParameterSet::newPSet)
    .def("dump", &PythonParameterSet::dump)
  ;

  boost::python::class_<PythonProcessDesc>("ProcessDesc", boost::python::init<>())
    .def(boost::python::init<std::string>())
    .def("newPSet", &PythonProcessDesc::newPSet)
    .def("dump", &PythonProcessDesc::dump)
  ;
}
#endif
