// python wrappers for CondDB
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"
#include "CondCore/IOVService/interface/IOVProxy.h"


#include <boost/python.hpp>

using namespace boost::python;

namespace {
//  exceptionTranslator(const edm::Exception & e)
//  {
//    PyErr_SetString(PyExc_RuntimeError, e.what());
//  }
  void exceptionTranslator(const std::exception & e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
}

BOOST_PYTHON_MODULE(pluginCondDBPyInterface) {
  
  class_<cond::IOVElement>("IOVElement", init<>())
    .def("since", &cond::IOVElement::since)
    .def("till", &cond::IOVElement::till)
    .def("payloadToken", &cond::IOVElement::payloadToken, return_value_policy<copy_const_reference>());
  
  class_<cond::IOVProxy>("IOV", init<>())
    .def("size", &cond::IOVProxy::size)
    .add_property("elements", range( &cond::IOVProxy::begin,  &cond::IOVProxy::end));
  
  
  class_<cond::FWIncantation>("FWIncantation", init<>());
  
  class_<cond::CondDB>("CondDB", init<>())
    .def("allTags", &cond::CondDB::allTags)
    .def("iov", &cond::CondDB::iov);
    .def("iovWithLib", &cond::CondDB::iovWithLib);
  
  class_<cond::RDBMS>("RDBMS", init<>())
    .def(init<std::string>())
    .def(init<std::string, std::string>())
    .def("getDB", &cond::RDBMS::getDB);


//  register_exception_translator<edm::Exception>(exceptionTranslator);
  register_exception_translator<std::exception>(exceptionTranslator);


}
