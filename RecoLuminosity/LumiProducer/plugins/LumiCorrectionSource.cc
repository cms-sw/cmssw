// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      LumiCorrectionSource
// 
/**\class LumiCorrectionSource LumiCorrectionSource.cc RecoLuminosity/LumiProducer/src/LumiCorrectionSource.cc
Description: A essource/esproducer for lumi correction factor and run parameters needed to deduce the corrections
      Author: Zhen Xie
*/

//#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralKernel/Context.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "CoralKernel/IPropertyManager.h"
#include "RelationalAccess/AuthenticationServiceException.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParamRcd.h"
#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include "RecoLuminosity/LumiProducer/interface/NormDML.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "LumiCorrectionSource.h"
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <boost/filesystem.hpp>
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
std::string 
LumiCorrectionSource::x2s(const XMLCh *toTranscode)const{
  std::string tmp(xercesc::XMLString::transcode(toTranscode));
  return tmp;
}

XMLCh*  
LumiCorrectionSource::s2x( const std::string& temp )const{
  XMLCh* buff = xercesc::XMLString::transcode(temp.c_str());    
  return  buff;
}

std::string
LumiCorrectionSource::toParentString(const xercesc::DOMNode &nodeToConvert)const{
  std::ostringstream oss;
  xercesc::DOMNodeList *childList = nodeToConvert.getChildNodes();

  unsigned int numNodes = childList->getLength ();
  for (unsigned int i = 0; i < numNodes; ++i){
    xercesc::DOMNode *childNode = childList->item(i);
    if (childNode->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)
      continue;
    xercesc::DOMElement *child = static_cast < xercesc::DOMElement *> (childNode);
    xercesc::DOMNamedNodeMap *attributes = child->getAttributes();
    unsigned int numAttributes = attributes->getLength ();
    for (unsigned int j = 0; j < numAttributes; ++j){
      xercesc::DOMNode *attributeNode = attributes->item(j);
      if (attributeNode->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
	continue;
      xercesc::DOMAttr *attribute = static_cast <xercesc::DOMAttr *> (attributeNode);
      
      oss << "(" << x2s(child->getTagName()) << 
	x2s(attribute->getName()) << "=" << 
	x2s(attribute->getValue()) << ")";
    }
  }
  return oss.str();
}

const std::string
LumiCorrectionSource::servletTranslation(const std::string& servlet) const{
  std::string frontierConnect;
  std::string realconnect;
  cms::concurrency::xercesInitialize();  
  std::unique_ptr< xercesc::XercesDOMParser > parser(new xercesc::XercesDOMParser);
  try{
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
    parser->setDoNamespaces(false);
    parser->parse(m_siteconfpath.c_str());
    xercesc::DOMDocument* doc=parser->getDocument();
    if(!doc){
      return "";
    }
   
    xercesc::DOMNodeList *frontierConnectList=doc->getElementsByTagName(s2x("frontier-connect"));
    if (frontierConnectList->getLength()>0){
      xercesc::DOMElement *frontierConnectElement=static_cast < xercesc::DOMElement *> (frontierConnectList->item (0));
      frontierConnect = toParentString(*frontierConnectElement);
    }
    // Replace the last component of every "serverurl=" piece (up to the
    //   next close-paren) with the servlet
    std::string::size_type nextparen = 0;
    std::string::size_type serverurl, lastslash;
    std::string complexstr = "";
    while ((serverurl=frontierConnect.find("(serverurl=", nextparen)) != std::string::npos){
      realconnect.append(frontierConnect, nextparen, serverurl - nextparen);
      nextparen=frontierConnect.find(')', serverurl);
      lastslash=frontierConnect.rfind('/', nextparen);
      realconnect.append(frontierConnect,serverurl,lastslash-serverurl+1);
      realconnect.append(servlet);
    }
    realconnect.append(frontierConnect, nextparen,frontierConnect.length()-nextparen);
  }catch(xercesc::DOMException &e){
  }
  return realconnect;
}

std::string 
LumiCorrectionSource::translateFrontierConnect(const std::string& connectStr){
  std::string result;
  const std::string fproto("frontier://");
  std::string::size_type startservlet=fproto.length();
  std::string::size_type endservlet=connectStr.find("(",startservlet);
  if(endservlet==std::string::npos){
    endservlet=connectStr.rfind('/',connectStr.length());
  }
  std::string servlet=connectStr.substr(startservlet,endservlet-startservlet);
  if( (!servlet.empty())&& (servlet.find_first_of(":/)[]")==std::string::npos)){
    if(servlet=="cms_conditions_data") servlet="";
    if(m_siteconfpath.length()==0){
      std::string url=(boost::filesystem::path("SITECONF")/boost::filesystem::path("local")/boost::filesystem::path("JobConfig")/boost::filesystem::path("site-local-config.xml")).string();
      char * tmp = getenv ("CMS_PATH");
      if(tmp){
	m_siteconfpath = (boost::filesystem::path(tmp)/boost::filesystem::path(url)).string();
      }
    }else{
      if(!boost::filesystem::exists(boost::filesystem::path(m_siteconfpath))){
	throw cms::Exception("Non existing path ")<<m_siteconfpath;
      }
      m_siteconfpath = (boost::filesystem::path(m_siteconfpath)/boost::filesystem::path("site-local-config.xml")).string();
    }
    result=fproto+servletTranslation(servlet)+connectStr.substr(endservlet);
  }
  return result;
}

LumiCorrectionSource::LumiCorrectionSource(const edm::ParameterSet& iConfig):m_connectStr(""),m_authfilename(""),m_datatag(""),m_globaltag(""),m_normtag(""),m_paramcachedrun(0),m_cachesize(0){
  setWhatProduced(this,&LumiCorrectionSource::produceLumiCorrectionParam);
  findingRecord<LumiCorrectionParamRcd>();
  std::string connectStr=iConfig.getParameter<std::string>("connect");
  std::string globaltag;
  if(iConfig.exists("globaltag")){
    m_globaltag=iConfig.getUntrackedParameter<std::string>("globaltag","");
  }else{
    m_normtag=iConfig.getUntrackedParameter<std::string>("normtag","");
  }
  m_datatag=iConfig.getUntrackedParameter<std::string>("datatag","");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",3);
  m_siteconfpath=iConfig.getUntrackedParameter<std::string>("siteconfpath","");
  const std::string fproto("frontier://");
  if(connectStr.substr(0,fproto.length())==fproto){
    m_connectStr=translateFrontierConnect(connectStr);
  }else if(connectStr.substr(0,11)=="sqlite_file"){
    m_connectStr=connectStr;
  }else{
    m_connectStr=connectStr;
    std::string authpath=iConfig.getUntrackedParameter<std::string>("authpath","");
    boost::filesystem::path boostAuthPath( authpath );
    if(boost::filesystem::is_directory(boostAuthPath)){
      boostAuthPath /= boost::filesystem::path("authentication.xml");      
    }
    m_authfilename = boostAuthPath.string();
  }
}

LumiCorrectionSource::ReturnParamType
LumiCorrectionSource::produceLumiCorrectionParam(const LumiCorrectionParamRcd&)  
{ 
  unsigned int currentrun=m_pcurrentTime->eventID().run();
  if(currentrun==0||currentrun==4294967295){ 
    return std::make_shared<const LumiCorrectionParam>();
  }
  if(m_paramcachedrun!=currentrun){//i'm in a new run
    fillparamcache(currentrun);//fill cache
  }else{ //i'm in an old run
    if(m_paramcache.find(currentrun)==m_paramcache.end()){//i'm not cached 
      fillparamcache(currentrun);// 
    }
  }
  if(m_paramcache.empty()){
    return std::make_shared<const LumiCorrectionParam>();
  }
  m_paramresult=m_paramcache[currentrun];
  if(m_paramresult.get()==nullptr){
    return std::make_shared<const LumiCorrectionParam>();
  }
  return m_paramresult;
}

void 
LumiCorrectionSource::setIntervalFor( 
    const edm::eventsetup::EventSetupRecordKey& iKey, 
    const edm::IOVSyncValue& iTime, 
    edm::ValidityInterval& oValidity ) {
  m_pcurrentTime=&iTime;
  oValidity.setFirst(iTime);
  oValidity.setLast(iTime);
}

void 
LumiCorrectionSource::reloadAuth(){
  //std::cout<<"old authfile "<<coral::Context::instance().PropertyManager().property("AuthenticationFile")->get()<<std::endl;
  coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(m_authfilename);
  coral::Context::instance().loadComponent("CORAL/Services/XMLAuthenticationService");
}

void
LumiCorrectionSource::fillparamcache(unsigned int runnumber){
  m_paramcache.clear();
  m_paramcachedrun=runnumber;
  if(!m_authfilename.empty()){
    coral::IHandle<coral::IAuthenticationService> authSvc=coral::Context::instance().query<coral::IAuthenticationService>();
    if( authSvc.isValid() ){
      try{
	authSvc->credentials( m_connectStr );
      }catch(const coral::UnknownConnectionException& er){
	reloadAuth();
      }
  }else{
      reloadAuth();
    }
  }
  coral::ConnectionService* mydbservice=new coral::ConnectionService;
  if(!m_globaltag.empty()){
    coral::ISessionProxy* gsession=mydbservice->connect(m_connectStr,coral::ReadOnly);
    gsession->transaction().start(true);
    parseGlobaltagForLumi(gsession->nominalSchema(),m_globaltag);
    gsession->transaction().commit();
    delete gsession;
  }
  coral::ISessionProxy* session=mydbservice->connect(m_connectStr,coral::ReadOnly);
  coral::ITypeConverter& tconverter=session->typeConverter();
  tconverter.setCppTypeForSqlType(std::string("float"),std::string("FLOAT(63)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned int"),std::string("NUMBER(10)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned short"),std::string("NUMBER(1)"));
  auto result = std::make_unique<LumiCorrectionParam>(LumiCorrectionParam::HF);
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    lumi::RevisionDML dml;
    unsigned long long tagid=0;
    if(m_datatag.empty()){
      tagid=dml.currentHFDataTagId(schema);//get datatag id
    }else{
      tagid=dml.HFDataTagIdByName(schema,m_datatag);
    }
    lumi::RevisionDML::DataID dataid=dml.dataIDForRun(schema,runnumber,tagid);//get data id
    unsigned int lumiid=dataid.lumi_id;
    if(lumiid==0){
      result->setNBX(0);
      std::shared_ptr<const LumiCorrectionParam> const_result = std::move(result);
      m_paramcache.insert(std::make_pair(runnumber,const_result));
      session->transaction().commit();
      delete session;
      delete mydbservice;
      return;
    }
    
    coral::AttributeList lumidataBindVariables;
    lumidataBindVariables.extend("dataid",typeid(unsigned long long));
    lumidataBindVariables["dataid"].data<unsigned long long>()=lumiid;   
    std::string conditionStr("DATA_ID=:dataid");
    coral::AttributeList lumiparamOutput;
    lumiparamOutput.extend("NCOLLIDINGBUNCHES",typeid(unsigned int));
    coral::IQuery* lumiparamQuery=schema.newQuery();
    lumiparamQuery->addToTableList(std::string("LUMIDATA"));
    lumiparamQuery->setCondition(conditionStr,lumidataBindVariables);
    lumiparamQuery->addToOutputList("NCOLLIDINGBUNCHES");
    lumiparamQuery->defineOutput(lumiparamOutput);
    coral::ICursor& lumiparamcursor=lumiparamQuery->execute();
    unsigned int ncollidingbx=0;
    while( lumiparamcursor.next() ){
      const coral::AttributeList& row=lumiparamcursor.currentRow();
      if(!row["NCOLLIDINGBUNCHES"].isNull()){
	ncollidingbx=row["NCOLLIDINGBUNCHES"].data<unsigned int>();
      }
      result->setNBX(ncollidingbx);
    }
    delete lumiparamQuery;
    lumi::NormDML normdml;
    unsigned long long normid=0;
    std::map<std::string,unsigned long long> normidmap;
    if (m_normtag.empty()){
      normdml.normIdByType(schema,normidmap,lumi::NormDML::HF,true);
      m_normtag=normidmap.begin()->first;
      normid=normidmap.begin()->second;
    }else{
      normid=normdml.normIdByName(schema,m_normtag);
    }

    std::map< unsigned int,lumi::NormDML::normData > normDataMap;
    normdml.normById(schema,normid,normDataMap); 
    
    std::map< unsigned int,lumi::NormDML::normData >::iterator normIt=--normDataMap.end();
    if(runnumber<normIt->first){
      normIt=normDataMap.upper_bound(runnumber);
      --normIt;
    }
    result->setNormtag(m_normtag);
    result->setcorrFunc(normIt->second.corrfunc);
    result->setnonlinearCoeff(normIt->second.coefficientmap);
    result->setafterglows(normIt->second.afterglows);
    result->setdescription(normIt->second.amodetag,normIt->second.beamegev);
    if(normIt->second.coefficientmap["DRIFT"]!=0.){
      float intglumi=fetchIntglumi(schema,runnumber);
      result->setintglumi(intglumi);
    }
    m_paramcache.insert(std::make_pair(runnumber,std::shared_ptr<LumiCorrectionParam>(std::move(result))));
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    delete session;
    delete mydbservice;
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  delete session;
  delete mydbservice;
}
void
LumiCorrectionSource::parseGlobaltagForLumi(coral::ISchema& schema,const std::string& globaltag){
  /**select i.pfn,i.tagname from TAGINVENTORY_TABLE i,TAGTREE_TABLE_GLOBALTAG v from i.tagid=v.tagid and i.recordname='LumiCorrectionParamRcd' **/
  std::string tagtreetabname("TAGTREE_TABLE_");
  tagtreetabname+=std::string(globaltag);
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList("TAGINVENTORY_TABLE","i");
  qHandle->addToTableList(tagtreetabname,"v");
  coral::AttributeList qResult;
  qResult.extend("pfn",typeid(std::string));
  qResult.extend("tagname",typeid(std::string));
  std::string conditionStr("v.tagid=i.tagid and i.recordname=:recordname");
  coral::AttributeList qCondition;
  qCondition.extend("recordname",typeid(std::string));
  qCondition["recordname"].data<std::string>()=std::string("LumiCorrectionParamRcd");
  qHandle->setCondition(conditionStr,qCondition);
  qHandle->addToOutputList("i.pfn");
  qHandle->addToOutputList("i.tagname");
  qHandle->defineOutput(qResult);
  coral::ICursor& iCursor=qHandle->execute();
  while( iCursor.next() ){
    const coral::AttributeList& row=iCursor.currentRow();
    std::string connectStr=row["pfn"].data<std::string>();
    const std::string fproto("frontier://");
    if(connectStr.substr(0,fproto.length())==fproto){
      m_connectStr=translateFrontierConnect(connectStr);
    }else{
      m_connectStr=connectStr;
    }
    m_normtag=row["tagname"].data<std::string>();
  }
  delete qHandle;
}
float 
LumiCorrectionSource::fetchIntglumi(coral::ISchema& schema,unsigned int runnumber){
  float result=0.;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList(lumi::LumiNames::intglumiv2TableName());
  coral::AttributeList qResult;
  qResult.extend("INTGLUMI",typeid(float));
  std::string conditionStr("RUNNUM=:runnumber");
  coral::AttributeList qCondition;
  qCondition.extend("runnumber",typeid(unsigned int));
  qCondition["runnumber"].data<unsigned int>()=runnumber;
  qHandle->setCondition(conditionStr,qCondition);
  qHandle->addToOutputList("INTGLUMI");
  qHandle->defineOutput(qResult);
  coral::ICursor& intglumiCursor=qHandle->execute();
  while( intglumiCursor.next() ){
    const coral::AttributeList& row=intglumiCursor.currentRow();
    if(!row["INTGLUMI"].isNull()){
      result=row["INTGLUMI"].data<float>();
    }
  }
  delete qHandle;
  return result;
}

LumiCorrectionSource::~LumiCorrectionSource(){}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(LumiCorrectionSource);
