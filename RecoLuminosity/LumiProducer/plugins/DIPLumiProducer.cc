// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      DIPLumiProducer
// 
/**\class DIPLumiProducer DIPLumiProducer.cc RecoLuminosity/LumiProducer/src/DIPLumiProducer.cc
Description: A essource/esproducer for lumi values from DIP via runtime logger DB
*/
// $Id: DIPLumiProducer.cc,v 1.10 2012/04/03 10:35:18 xiezhen Exp $

//#include <memory>
//#include "boost/shared_ptr.hpp"
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
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLumiDetail.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLuminosityRcd.h"
#include "DIPLumiProducer.h"
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <iterator>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

DIPLumiProducer::DIPLumiProducer(const edm::ParameterSet& iConfig):m_connectStr(""),m_summarycachedrun(0),m_detailcachedrun(0),m_cachesize(0){
  setWhatProduced(this,&DIPLumiProducer::produceSummary);
  setWhatProduced(this,&DIPLumiProducer::produceDetail);
  findingRecord<DIPLuminosityRcd>();
  m_connectStr=iConfig.getParameter<std::string>("connect");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",5);
}

DIPLumiProducer::ReturnSummaryType
DIPLumiProducer::produceSummary(const DIPLuminosityRcd&)  
{ 
  unsigned int currentrun=m_pcurrentTime->eventID().run();
  unsigned int currentls=m_pcurrentTime->luminosityBlockNumber();
  if(currentls==0||currentls==4294967295){ 
    return  boost::shared_ptr<DIPLumiSummary>(new DIPLumiSummary());
  }
  if(m_summarycachedrun!=currentrun){//i'm in a new run
    fillsummarycache(currentrun,currentls);//starting ls
  }else{ //i'm in an old run
    if(m_summarycache.find(currentls)==m_summarycache.end()){//i'm in a uncached 
      fillsummarycache(currentrun,currentls);//cache all ls>=currentls 
      if(m_summarycache.find(currentls)==m_summarycache.end()){
	
	return boost::shared_ptr<DIPLumiSummary>(new DIPLumiSummary()); //really no data found
      }
    }
  }
  m_summaryresult=m_summarycache[currentls];
  //std::cout<<*m_summaryresult<<std::endl;
  if(m_summaryresult.get()==0){
    return boost::shared_ptr<DIPLumiSummary>(new DIPLumiSummary());
  }
  return m_summaryresult;
}
DIPLumiProducer::ReturnDetailType
DIPLumiProducer::produceDetail(const DIPLuminosityRcd&)  
{ 
  unsigned int currentrun=m_pcurrentTime->eventID().run();
  unsigned int currentls=m_pcurrentTime->luminosityBlockNumber();
  if(currentls==0||currentls==4294967295){ 
    return  boost::shared_ptr<DIPLumiDetail>(new DIPLumiDetail());
  }
  if(m_detailcachedrun!=currentrun){//i'm in a new run
    filldetailcache(currentrun,currentls);//starting ls
  }else{ //i'm in an known run
    if(m_detailcache.find(currentls)==m_detailcache.end()){//i'm in a unknown ls
      filldetailcache(currentrun,currentls);//cache all ls>=currentls 
      if(m_detailcache.find(currentls)==m_detailcache.end()){
	return boost::shared_ptr<DIPLumiDetail>(new DIPLumiDetail()); //really no data found
      }
    }
  }
  m_detailresult=m_detailcache[currentls];
  if(m_detailresult.get()==0){
    return boost::shared_ptr<DIPLumiDetail>(new DIPLumiDetail());
  }
  return m_detailresult;
}

void 
DIPLumiProducer::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, 
				 const edm::IOVSyncValue& iTime, 
				 edm::ValidityInterval& oValidity ) {
  m_pcurrentTime=&iTime;
  oValidity.setFirst(iTime);
  oValidity.setLast(iTime);
}

void
DIPLumiProducer::fillsummarycache(unsigned int runnumber,unsigned int startlsnum){
  m_summarycache.clear();
  m_summarycachedrun=runnumber;
  //
  //queries once per cache refill
  //
  //select lumisection,instlumi,delivlumi,livelumi from cms_runtime_logger.lumi_sections where lumisection>=:lsmin and lumisection<:lsmax and runnumber=:runnumber;
  //
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  coral::ITypeConverter& tconverter=session->typeConverter();
  tconverter.setCppTypeForSqlType(std::string("float"),std::string("FLOAT(63)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned int"),std::string("NUMBER(10)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned short"),std::string("NUMBER(1)"));
  unsigned int lsmin=1;
  unsigned int lsmax=0;
  if(startlsnum!=0){
    lsmin=startlsnum;
  }
  if(m_cachesize!=0){
    lsmax=lsmin+m_cachesize;
  }
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    coral::AttributeList lumisummaryBindVariables;
    lumisummaryBindVariables.extend("lsmin",typeid(unsigned int));
    lumisummaryBindVariables.extend("runnumber",typeid(unsigned int));
    lumisummaryBindVariables["runnumber"].data<unsigned int>()=m_summarycachedrun;
    lumisummaryBindVariables["lsmin"].data<unsigned int>()=lsmin;
    std::string conditionStr(" RUNNUMBER=:runnumber AND LUMISECTION>=:lsmin ");
    coral::AttributeList lumisummaryOutput;
    lumisummaryOutput.extend("LUMISECTION",typeid(unsigned int));
    lumisummaryOutput.extend("INSTLUMI",typeid(float));
    lumisummaryOutput.extend("DELIVLUMISECTION",typeid(float));
    lumisummaryOutput.extend("LIVELUMISECTION",typeid(float));
    lumisummaryOutput.extend("CMS_ACTIVE",typeid(unsigned short));
    if(m_cachesize!=0){
      lumisummaryBindVariables.extend("lsmax",typeid(unsigned int));
      conditionStr=conditionStr+"AND LUMISECTION<=:lsmax";
      lumisummaryBindVariables["lsmax"].data<unsigned int>()=lsmax;      
    }
    coral::IQuery* lumisummaryQuery=schema.newQuery();
    lumisummaryQuery->addToTableList(std::string("LUMI_SECTIONS"));
    lumisummaryQuery->addToOutputList("LUMISECTION");
    lumisummaryQuery->addToOutputList("INSTLUMI");
    lumisummaryQuery->addToOutputList("DELIVLUMISECTION");
    lumisummaryQuery->addToOutputList("LIVELUMISECTION");
    lumisummaryQuery->addToOutputList("CMS_ACTIVE");
    lumisummaryQuery->setCondition(conditionStr,lumisummaryBindVariables);
    lumisummaryQuery->defineOutput(lumisummaryOutput);
    coral::ICursor& lumisummarycursor=lumisummaryQuery->execute();
    unsigned int rowcounter=0;
    while( lumisummarycursor.next() ){
      const coral::AttributeList& row=lumisummarycursor.currentRow();
      unsigned int lsnum=row["LUMISECTION"].data<unsigned int>();
      float instlumi=row["INSTLUMI"].data<float>();//Hz/ub
      float intgdellumi=row["DELIVLUMISECTION"].data<float>()*1000.0;//convert to /ub
      float intgreclumi=row["LIVELUMISECTION"].data<float>()*1000.0;//convert to /ub
      unsigned short cmsalive=row["CMS_ACTIVE"].data<unsigned short>();
      boost::shared_ptr<DIPLumiSummary> tmpls(new DIPLumiSummary(instlumi,intgdellumi,intgreclumi,cmsalive));
      m_summarycache.insert(std::make_pair(lsnum,tmpls));
      ++rowcounter;
    }
    //if (rowcounter==0){
    //  m_isNullRun=true;
    //}
    delete lumisummaryQuery;
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  mydbservice->disconnect(session);
}
void
DIPLumiProducer::filldetailcache(unsigned int runnumber,unsigned int startlsnum){
  m_detailcache.clear();
  m_detailcachedrun=runnumber;
  //
  //queries once per cache refill
  //
  //select lumisection,bunch,bunchlumi from cms_runtime_logger.bunch_lumi_sections where lumisection>=:lsmin and lumisection<:lsmax and runnumber=:runnumber;
  //
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  coral::ITypeConverter& tconverter=session->typeConverter();
  tconverter.setCppTypeForSqlType(std::string("float"),std::string("FLOAT(63)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned int"),std::string("NUMBER(10)"));
  unsigned int lsmin=1;
  unsigned int lsmax=0;
  if(startlsnum!=0){
    lsmin=startlsnum;
  }
  lsmax=lsmin;
  if(m_cachesize!=0){
    lsmax=lsmin+m_cachesize;
  }
  for(unsigned int i=lsmin;i<=lsmax;++i){
    m_detailcache.insert(std::make_pair(i,boost::shared_ptr<DIPLumiDetail>(new DIPLumiDetail)));
  }
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    coral::AttributeList lumidetailBindVariables;
    lumidetailBindVariables.extend("lsmin",typeid(unsigned int));
    lumidetailBindVariables.extend("runnumber",typeid(unsigned int));
    lumidetailBindVariables["runnumber"].data<unsigned int>()=m_detailcachedrun;
    lumidetailBindVariables["lsmin"].data<unsigned int>()=lsmin;
    std::string conditionStr(" RUNNUMBER=:runnumber AND LUMISECTION>=:lsmin AND BUNCHLUMI>0 ");
    coral::AttributeList lumidetailOutput;
    lumidetailOutput.extend("LUMISECTION",typeid(unsigned int));
    lumidetailOutput.extend("BUNCH",typeid(unsigned int));
    lumidetailOutput.extend("BUNCHLUMI",typeid(float));
    if(m_cachesize!=0){
      lumidetailBindVariables.extend("lsmax",typeid(unsigned int));
      conditionStr=conditionStr+"AND LUMISECTION<=:lsmax";
      lumidetailBindVariables["lsmax"].data<unsigned int>()=lsmax;      
    }
    coral::IQuery* lumidetailQuery=schema.newQuery();
    lumidetailQuery->addToTableList(std::string("BUNCH_LUMI_SECTIONS"));
    lumidetailQuery->addToOutputList("LUMISECTION");
    lumidetailQuery->addToOutputList("BUNCH");
    lumidetailQuery->addToOutputList("BUNCHLUMI");
    lumidetailQuery->setCondition(conditionStr,lumidetailBindVariables);
    lumidetailQuery->defineOutput(lumidetailOutput);
    coral::ICursor& lumidetailcursor=lumidetailQuery->execute();
    unsigned int rowcounter=0;
    while( lumidetailcursor.next() ){
      const coral::AttributeList& row=lumidetailcursor.currentRow();
      unsigned int lsnum=row["LUMISECTION"].data<unsigned int>();
      unsigned int bxidx=row["BUNCH"].data<unsigned int>();
      float bxlumi=row["BUNCHLUMI"].data<float>();//Hz/ub
      m_detailcache[lsnum]->fillbxdata(bxidx,bxlumi);
      ++rowcounter;
    }
    //if (rowcounter==0){
    //  m_isNullRun=true;
    //}
    delete lumidetailQuery;
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  mydbservice->disconnect(session);
}
DIPLumiProducer::~DIPLumiProducer(){}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(DIPLumiProducer);
