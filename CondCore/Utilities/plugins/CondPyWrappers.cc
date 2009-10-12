// python wrappers for CondDB
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"


#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondCore/DBCommon/interface/ClassID.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "StorageSvc/DbReflex.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

#include<iostream>

namespace {

  // find and return
  boost::shared_ptr<cond::ClassInfo> pyInfo(std::string const & token) {
    //    topinit();    
    static std::string const prefix = cond::idCategories::pythonIDCategory + "/";
    std::string pluginName = prefix + cond::classID(token);
    return boost::shared_ptr<cond::ClassInfo>(cond::ClassInfoFactory::get()->create(pluginName));
  }
  
  std::string moduleNameByTag(cond::CondDB & db, std::string const & tag) {
    //topinit();    
    cond::IOVProxy iov = db.iov(tag);
    if (0==iov.size()) return std::string();
    return pyInfo(iov.begin()->wrapperToken())->resource();
  }

  std::string moduleNameByToken(std::string const & token) {
    //topinit();    
    if (token.empty()) return std::string();
    return pyInfo(token)->resource();
  }
  
  std::string moduleName(cond::CondDB & db, std::string const & ss) {
    //topinit();    
    //assume tags never start with '['
    if (ss[0]=='[') return moduleNameByToken(ss);
    return  moduleNameByTag(db,ss);
  }


//  exceptionTranslator(const edm::Exception & e)
//  {
//    PyErr_SetString(PyExc_RuntimeError, e.what());
//  }
  void exceptionTranslator(const std::exception & e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }


    boost::python::tuple
    getState(cond::LogDBEntry& l)
    {
      return boost::python::make_tuple(
				       l.logId,
				       l.destinationDB,
				       l.provenance,
				       l.usertext,
				       l.iovtag,
				       l.iovtimetype,
				       l.payloadIdx,
				       l.payloadName,
				       l.payloadToken,
				       l.payloadContainer,
				       l.exectime,
				       l.execmessage
				       );
    }
 

  void append2VS(std::vector<std::string> & v, std::string s) {
    v.push_back(s);
  }

}

BOOST_PYTHON_MODULE(pluginCondDBPyInterface) {

  def("append2VS",&append2VS);

  class_<cond::LogDBEntry>("LogDBEntry")
    .def("getState",getState)
    .def_readonly("logId",  &cond::LogDBEntry::logId)
    .def_readonly("destinationDB",   &cond::LogDBEntry::destinationDB)   
    .def_readonly("provenance",  &cond::LogDBEntry::provenance)
    .def_readonly("usertext",  &cond::LogDBEntry::usertext)
    .def_readonly("iovtag",  &cond::LogDBEntry::iovtag)
    .def_readonly("iovtimetype",   &cond::LogDBEntry::iovtimetype)
    .def_readonly("payloadIdx",   &cond::LogDBEntry::payloadIdx)
    .def_readonly("payloadName",   &cond::LogDBEntry::payloadName)
    .def_readonly("payloadToken",   &cond::LogDBEntry::payloadToken)
    .def_readonly("payloadContainer",   &cond::LogDBEntry::payloadContainer)
    .def_readonly("exectime",   &cond::LogDBEntry::exectime)
    .def_readonly("execmessage",  &cond::LogDBEntry::execmessage)
    ;

  class_<std::vector<std::string> >("VString")
    .def(vector_indexing_suite<std::vector<std::string> >())
    ;

  class_<std::vector<float> >("VFloat")
    .def(vector_indexing_suite<std::vector<float> >())
    ;

  class_<std::vector<int> >("VInt")
    .def(vector_indexing_suite<std::vector<int> >())
    ;
   
  class_<cond::IOVElementProxy>("IOVElement", init<>())
    .def("since", &cond::IOVElementProxy::since)
    .def("till", &cond::IOVElementProxy::till)
    .def("payloadToken", &cond::IOVElementProxy::wrapperToken, return_value_policy<copy_const_reference>())
    ;
  
  enum_<cond::TimeType>("timetype")
    .value("runnumber",cond::runnumber)
    .value("timestamp",cond::timestamp)
    .value("lumiid",cond::lumiid)
    .value("userid",cond::userid)
    ;
  
  class_<cond::IOVProxy>("IOV", init<>())
    .def("size", &cond::IOVProxy::size)
    .def("setRange", &cond::IOVProxy::setRange)
    .def("head", &cond::IOVProxy::head)
    .def("tail", &cond::IOVProxy::tail)
    .def("timetype", &cond::IOVProxy::timetype)
    .add_property("elements", range( &cond::IOVProxy::begin,  &cond::IOVProxy::end))
    ;
  
  
  class_<cond::FWIncantation>("FWIncantation", init<>());
  
  class_<cond::CondDB>("CondDB", init<>())
    .def("allTags", &cond::CondDB::allTags)
    .def("iov", &cond::CondDB::iov)
    .def("iovWithLib", &cond::CondDB::iovWithLib)
    .def("payLoad", &cond::CondDB::payLoad)
    .def("moduleName",moduleName)
    .def("lastLogEntry", &cond::CondDB::lastLogEntry)
    .def("lastLogEntryOK", &cond::CondDB::lastLogEntryOK)
    ;
  

  class_<cond::RDBMS>("RDBMS", init<>())
    .def(init<std::string>())
    .def(init<std::string, std::string>())
    .def("setLogger",&cond::RDBMS::setLogger)
    .def("getDB", &cond::RDBMS::getDB)
    ;


//  register_exception_translator<edm::Exception>(exceptionTranslator);
  register_exception_translator<std::exception>(exceptionTranslator);


}
