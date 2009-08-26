#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/PythonParameterSet/interface/BoostPython.h"

#include <boost/cstdint.hpp>

#include <string>

using namespace boost::python;



BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{
  class_<edm::InputTag>("InputTag", init<std::string>())
      .def(init<std::string, std::string, std::string>())
     .def(init<std::string, std::string>())
      .def("label",    &edm::InputTag::label, return_value_policy<copy_const_reference>())
      .def("instance", &edm::InputTag::instance, return_value_policy<copy_const_reference>())
      .def("process",  &edm::InputTag::process, return_value_policy<copy_const_reference>())
  ;

   class_<edm::ESInputTag>("ESInputTag", init<std::string>())
   .def(init<std::string, std::string>())
   .def("module",    &edm::ESInputTag::module, return_value_policy<copy_const_reference>())
   .def("data",  &edm::ESInputTag::data, return_value_policy<copy_const_reference>())
   ;
   
   class_<edm::EventID>("EventID", init<unsigned int, unsigned int>())
      .def("run",   &edm::EventID::run)
      .def("event", &edm::EventID::event)
  ;

  class_<edm::LuminosityBlockID>("LuminosityBlockID", init<unsigned int, unsigned int>())
      .def("run",    &edm::LuminosityBlockID::run)
      .def("luminosityBlock", &edm::LuminosityBlockID::luminosityBlock)
  ;

  class_<edm::FileInPath>("FileInPath", init<std::string>())
      .def("fullPath",     &edm::FileInPath::fullPath)
      .def("relativePath", &edm::FileInPath::relativePath)
      .def("isLocal",      &edm::FileInPath::isLocal)
  ;

  class_<edm::LuminosityBlockRange>("LuminosityBlockRange", init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def("start",    &edm::LuminosityBlockRange::startRun)
      .def("startSub", &edm::LuminosityBlockRange::startLumi)
      .def("end",      &edm::LuminosityBlockRange::endRun)
      .def("endSub",   &edm::LuminosityBlockRange::endLumi)
  ;

  class_<edm::EventRange>("EventRange", init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def("start",    &edm::EventRange::startRun)
      .def("startSub", &edm::EventRange::startEvent)
      .def("end",      &edm::EventRange::endRun)
      .def("endSub",   &edm::EventRange::endEvent)
  ;



  class_<PythonParameterSet>("ParameterSet")
    .def("addInt32", &PythonParameterSet::addParameter<int>)
    .def("getInt32", &PythonParameterSet::getParameter<int>)
    .def("addVInt32", &PythonParameterSet::addParameters<int>)
    .def("getVInt32", &PythonParameterSet::getParameters<int>)
    .def("addUInt32", &PythonParameterSet::addParameter<unsigned int>)
    .def("getUInt32", &PythonParameterSet::getParameter<unsigned int>)
    .def("addVUInt32", &PythonParameterSet::addParameters<unsigned int>)
    .def("getVUInt32", &PythonParameterSet::getParameters<unsigned int>)
    .def("addInt64", &PythonParameterSet::addParameter<boost::int64_t>)
    .def("getInt64", &PythonParameterSet::getParameter<boost::int64_t>)
    .def("addUInt64", &PythonParameterSet::addParameter<boost::uint64_t>)
    .def("getUInt64", &PythonParameterSet::getParameter<boost::uint64_t>)
    .def("addVInt64", &PythonParameterSet::addParameters<boost::int64_t>)
    .def("getVInt64", &PythonParameterSet::getParameters<boost::int64_t>)
    .def("addVUInt64", &PythonParameterSet::addParameters<boost::uint64_t>)
    .def("getVUInt64", &PythonParameterSet::getParameters<boost::uint64_t>)
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


  class_<PythonProcessDesc>("ProcessDesc", init<>())
    .def(init<std::string>())
    .def("addService", &PythonProcessDesc::addService)
    .def("newPSet", &PythonProcessDesc::newPSet)
    .def("dump", &PythonProcessDesc::dump)
  ;

}


