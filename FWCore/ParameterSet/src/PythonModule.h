#include "FWCore/ParameterSet/interface/PythonParseTree.h"
#include "FWCore/ParameterSet/interface/PythonParameterSet.h"
#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include <boost/python.hpp>

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

  class_<edm::EventID>("EventID", init<unsigned int, unsigned int>())
      .def("run",    &edm::EventID::run)
      .def("event", &edm::EventID::event)
  ;


  class_<edm::FileInPath>("FileInPath", init<std::string>())
      .def("fullPath",     &edm::FileInPath::fullPath)
      .def("relativePath", &edm::FileInPath::relativePath)
      .def("isLocal",      &edm::FileInPath::isLocal)
  ;


  class_<PythonParseTree>("PythonParseTree", init<std::string>())
      .def("modules",       &PythonParseTree::modules)
      .def("modulesOfType", &PythonParseTree::modulesOfType)
      .def("process",       &PythonParseTree::process)
      .def("replaceValue",  &PythonParseTree::replaceValue)
      .def("replaceValues", &PythonParseTree::replaceValues)
      .def("dump",          &PythonParseTree::dump)
      .def("typeOf",        &PythonParseTree::typeOf)
      .def("value",         &PythonParseTree::value)
      .def("values",        &PythonParseTree::values)
      .def("children",      &PythonParseTree::children)
      .def("dumpTree",      &PythonParseTree::dumpTree)
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
    .def("addEventID", &PythonParameterSet::addParameter<edm::EventID>)
    .def("getEventID", &PythonParameterSet::getParameter<edm::EventID>)
    .def("addEventID", &PythonParameterSet::addParameters<edm::EventID>)
    .def("getEventID", &PythonParameterSet::getParameters<edm::EventID>)
    .def("addPSet", &PythonParameterSet::addPSet)
    .def("getPSet", &PythonParameterSet::getPSet)
    .def("addVPSet", &PythonParameterSet::addVPSet)
    .def("getVPSet", &PythonParameterSet::getVPSet)
    .def("addFileInPath", &PythonParameterSet::addParameter<edm::FileInPath>)
    .def("getFileInPath", &PythonParameterSet::getParameter<edm::FileInPath>)
    .def("newInputTag", &PythonParameterSet::newInputTag)
    .def("newEventID", &PythonParameterSet::newEventID)
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

   register_exception_translator<edm::Exception>(PythonParseTree::exceptionTranslator);
}


