#include "FWCore/ParameterSet/interface/PythonParseTree.h"
#include "FWCore/ParameterSet/interface/PythonParameterSet.h"
#include <boost/python.hpp>

using namespace boost::python;



BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{

   class_<PythonParseTree>("PythonParseTree", init<std::string>())
       .def("modules",       &PythonParseTree::modules)
       .def("modulesOfType", &PythonParseTree::modulesOfType)
       .def("process",       &PythonParseTree::process)
       .def("replaceValue",  &PythonParseTree::replaceValue)
       .def("replaceValues", &PythonParseTree::replaceValues)
       .def("dump",          &PythonParseTree::dump)
       .def("typeOf",         &PythonParseTree::typeOf)
       .def("value",         &PythonParseTree::value)
       .def("values",        &PythonParseTree::values)
       .def("children",      &PythonParseTree::children)
       .def("dumpTree",      &PythonParseTree::dumpTree)
   ;

  class_<PythonParameterSet>("PythonParameterSet")
    .def("addInt32", &PythonParameterSet::addParameter<int>)
    .def("getInt32", &PythonParameterSet::getParameter<int>)
    .def("addVInt32", &PythonParameterSet::addParameters<int>)
    .def("getVInt32", &PythonParameterSet::getParameters<int>)
    .def("addDouble", &PythonParameterSet::addParameter<double>)
    .def("getDouble", &PythonParameterSet::getParameter<double>)
    .def("addVDouble", &PythonParameterSet::addParameters<double>)
    .def("getVDouble", &PythonParameterSet::getParameters<double>)
    .def("addString", &PythonParameterSet::addParameter<std::string>)
    .def("getString", &PythonParameterSet::getParameter<std::string>)
    .def("addVString", &PythonParameterSet::addParameters<std::string>)
    .def("getVString", &PythonParameterSet::getParameters<std::string>)
//    .def("addInputTag", &PythonParameterSet::addParameter<edm::InputTag>)
//    .def("getInputTag", &PythonParameterSet::getParameter<edm::InputTag>)
//    .def("addVInputTag", &PythonParameterSet::addParameters<edm::InputTag>)
//    .def("getVInt32", &PythonParameterSet::getParameters<edm::InputTag>)
//    .def("addPSet", &PythonParameterSet::addParameter<int>)
//    .def("getPSet", &PythonParameterSet::getParameter<int>)
//    .def("addVPSet", &PythonParameterSet::addParameters<int>)
//    .def("getVPSet", &PythonParameterSet::getParameters<int>)



  ;
     
   register_exception_translator<edm::Exception>(PythonParseTree::exceptionTranslator);
}


