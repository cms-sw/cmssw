// python wrappers for CondDB
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondCore/DBCommon/interface/ClassID.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "StorageSvc/DbReflex.h"

#include "FWCore/PluginManager/interface/PluginManager.h"


#include <boost/python.hpp>

using namespace boost::python;

#include<iostream>

namespace {

  // find and return
  boost::shared_ptr<cond::ClassInfo> pyInfo(std::string const & token) {
    static std::string const prefix = cond::idCategories::pythonIDCategory + "/";
    std::string pluginName = prefix + cond::classID(token);
    return boost::shared_ptr<cond::ClassInfo>(cond::ClassInfoFactory::get()->create(pluginName));
  }
  
  std::string moduleName(cond::CondDB & db, std::string const & tag) {
    cond::IOVProxy iov = db.iov(tag);
    if (0==iov.size()) return std::string();
    return pyInfo(iov.begin()->payloadToken())->resource();
  }
  

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
    .def("iov", &cond::CondDB::iov)
    .def("iovWithLib", &cond::CondDB::iovWithLib)
    .def("moduleName",moduleName);
  

  class_<cond::RDBMS>("RDBMS", init<>())
    .def(init<std::string>())
    .def(init<std::string, std::string>())
    .def("getDB", &cond::RDBMS::getDB);


//  register_exception_translator<edm::Exception>(exceptionTranslator);
  register_exception_translator<std::exception>(exceptionTranslator);


}
