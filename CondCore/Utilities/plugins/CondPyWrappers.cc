// python wrappers for CondDB
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondFormats/Common/interface/TimeConversions.h"


#include "CondCore/DBCommon/interface/ClassID.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

#include<iostream>

namespace {
  
  // decode token
  std::string classID(std::string const & token) {
    static std::string const clid("CLID=");
    std::string::size_type s = token.find(clid) + clid.size();
    std::string::size_type e = token.find(']',s);
    return token.substr(s,e-s);

  }
  // find and return
  boost::shared_ptr<cond::ClassInfo> pyInfo(std::string const & token) {
    //    topinit();    
    static std::string const prefix = cond::idCategories::pythonIDCategory + "/";
    std::string pluginName = prefix + classID(token);
    return boost::shared_ptr<cond::ClassInfo>(cond::ClassInfoFactory::get()->create(pluginName));
  }
  
  std::string moduleNameByTag(cond::CondDB & db, std::string const & tag) {
    //topinit();    
    cond::IOVProxy iov = db.iov(tag);
    if (0==iov.size()) return std::string();
    return pyInfo(iov.begin()->token())->resource();
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
    getLogState(cond::LogDBEntry& l)
    {
      return boost::python::make_tuple(
				       l.logId,
				       l.destinationDB,
				       l.provenance,
				       l.usertext,
				       l.iovtag,
				       l.iovtimetype,
				       l.payloadIdx,
				       l.payloadClass,
				       l.payloadToken,
				       l.exectime,
				       l.execmessage
				       );
    }
 
    boost::python::tuple
    getTagState(cond::TagMetadata& l)
    {
      return boost::python::make_tuple(
				       l.tag,
				       l.pfn,
				       l.recordname,
				       l.labelname,	       
				       l.objectname
				       );
    }


  void append2VS(std::vector<std::string> & v, std::string s) {
    v.push_back(s);
  }

  boost::python::tuple unpackTime(cond::Time_t iValue) {
    cond::UnpackedTime l = cond::time::unpack(iValue);
    return boost::python::make_tuple(l.first,l.second);
  }

}

BOOST_PYTHON_MODULE(pluginCondDBPyInterface) {

  def("append2VS",&append2VS);

  def("unpackTime",&unpackTime);

  class_<cond::LogDBEntry>("LogDBEntry")
    .def("getState",getLogState)
    .def_readonly("logId",  &cond::LogDBEntry::logId)
    .def_readonly("destinationDB",   &cond::LogDBEntry::destinationDB)   
    .def_readonly("provenance",  &cond::LogDBEntry::provenance)
    .def_readonly("usertext",  &cond::LogDBEntry::usertext)
    .def_readonly("iovtag",  &cond::LogDBEntry::iovtag)
    .def_readonly("iovtimetype",   &cond::LogDBEntry::iovtimetype)
    .def_readonly("payloadIdx",   &cond::LogDBEntry::payloadIdx)
    .def_readonly("payloadClass",   &cond::LogDBEntry::payloadClass)
    .def_readonly("payloadToken",   &cond::LogDBEntry::payloadToken)
    .def_readonly("exectime",   &cond::LogDBEntry::exectime)
    .def_readonly("execmessage",  &cond::LogDBEntry::execmessage)
    ;
  
  class_<cond::TagMetadata>("TagEntry")
    .def("getState",getTagState)
    .def_readonly("tag", &cond::TagMetadata::tag)
    .def_readonly("pfn", &cond::TagMetadata::pfn)
    .def_readonly("record", &cond::TagMetadata::recordname)
    .def_readonly("label", &cond::TagMetadata::labelname)
    .def_readonly("object", &cond::TagMetadata::objectname)
    ;
  
  class_<cond::GlobalTag >("GlobalTag", init<>())
    .def("size", &cond::GlobalTag::size)
    .add_property("elements",  boost::python::range( &cond::GlobalTag::begin,  &cond::GlobalTag::end))
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
    .def("payloadToken", &cond::IOVElementProxy::token, return_value_policy<copy_const_reference>())
    ;
  
  enum_<cond::TimeType>("timetype")
    .value("runnumber",cond::runnumber)
    .value("timestamp",cond::timestamp)
    .value("lumiid",cond::lumiid)
    .value("userid",cond::userid)
    ;
  
  class_<cond::IOVProxy>("IOV", init<>())
    .def("size", &cond::IOVProxy::size)
    .def("resetRange", &cond::IOVProxy::resetRange)
    .def("setRange", &cond::IOVProxy::setRange)
    .def("head", &cond::IOVProxy::head)
    .def("tail", &cond::IOVProxy::tail)
    .def("timetype", &cond::IOVProxy::timetype)
    .def("comment", &cond::IOVProxy::comment)
    .def("revision",&cond::IOVProxy::revision)
    .def("timestamp",&cond::IOVProxy::timestamp)
    .add_property("elements", boost::python::range( &cond::IOVProxy::begin,  &cond::IOVProxy::end))
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
    .def("getReadOnlyDB", &cond::RDBMS::getReadOnlyDB)
    .def("globalTag",  &cond::RDBMS::globalTag, return_value_policy<copy_const_reference>())
    ;


//  register_exception_translator<edm::Exception>(exceptionTranslator);
  register_exception_translator<std::exception>(exceptionTranslator);


}
