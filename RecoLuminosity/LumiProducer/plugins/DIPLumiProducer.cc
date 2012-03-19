// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      DIPLumiProducer
// 
/**\class DIPLumiProducer DIPLumiProducer.cc RecoLuminosity/LumiProducer/src/DIPLumiProducer.cc
Description: A essource/esproducer for lumi values from DIP via runtime logger DB
*/
// $Id: DIPLumiProducer.cc,v 1.1 2012/03/16 21:32:59 xiezhen Exp $

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
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummaryRcd.h"
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

DIPLumiProducer::DIPLumiProducer(const edm::ParameterSet& iConfig):m_connectStr(""),m_cachedrun(0),m_cachesize(0){
  setWhatProduced(this);
  findingRecord<DIPLumiSummaryRcd>();
  m_connectStr=iConfig.getParameter<std::string>("connect");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",0);
}

DIPLumiProducer::ReturnType
DIPLumiProducer::produce(const DIPLumiSummaryRcd&)  
{ 
  return m_result;
}

void 
DIPLumiProducer::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, 
				 const edm::IOVSyncValue& iTime, 
				 edm::ValidityInterval& oValidity ) {
  oValidity=edm::ValidityInterval::invalidInterval();//default
  unsigned int currentrun=iTime.eventID().run();
  unsigned int currentls=iTime.luminosityBlockNumber();
  std::cout<<"setIntervalFor run "<<currentrun<<", ls "<<currentls<<std::endl;
  std::cout<<"cached run "<<m_cachedrun<<std::endl;
  if(currentls==0||currentls==4294967295)return;//a fake setIntervalFor
  if(m_cachedrun!=currentrun){//found a new run
    clearcache();
    m_cachedrun=currentrun;
    fillcache(currentrun,0);//starting ls
    m_result=m_lscache[currentls];//copy construct
  }else{
    if(m_lscache.find(currentls)==m_lscache.end()){//if ls not cached
      fillcache(currentrun,currentls);//cache all ls>=currentls for this run
      if(m_lscache.find(currentls)==m_lscache.end()){
	std::cout<<"really no data found "<<std::endl;
	return;
      }
    }
    m_result=m_lscache[currentls];//copy construct
  }
  oValidity.setFirst(iTime);
  oValidity.setLast(iTime);
}

void
DIPLumiProducer::fillcache(unsigned int runnumber,unsigned int startlsnum){
  std::cout<<"fillcache cached run: "<<m_cachedrun<<" tofill "<<runnumber<<" , "<<startlsnum<<std::endl;
  m_lscache.clear();
  //
  //queries once per cache refill
  //
  //select lumisection,startorbit,instlumi,delivlumi,livelumi,deadtime from cms_runtime_logger.lumi_sections where lumisection>=:lsmin and lumisection<:lsmax and runnumber=:runnumber;
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
    lumisummaryBindVariables.extend("lsmax",typeid(unsigned int));
    lumisummaryBindVariables.extend("runnumber",typeid(unsigned int));
    lumisummaryBindVariables["runnumber"].data<unsigned int>()=m_cachedrun;
    lumisummaryBindVariables["lsmin"].data<unsigned int>()=lsmin;
    std::string conditionStr(" RUNNUMBER=:runnumber AND CMSLSNUM>=:lsmin ");
    coral::AttributeList lumisummaryOutput;
    lumisummaryOutput.extend("LUMISECTION",typeid(unsigned int));
    lumisummaryOutput.extend("INSTLUMI",typeid(float));
    lumisummaryOutput.extend("DELIVLUMI",typeid(float));
    lumisummaryOutput.extend("LIVELUMI",typeid(float));
    lumisummaryOutput.extend("DEADTIME",typeid(float));
    if(m_cachesize!=0){
      conditionStr=conditionStr+"AND CMSLSNUM<:lsmax";
      lumisummaryBindVariables["lsmax"].data<unsigned int>()=lsmax;      
    }
    coral::IQuery* lumisummaryQuery=schema.newQuery();
    lumisummaryQuery->addToTableList(std::string("LUMI_SECTIONS"));
    lumisummaryQuery->addToOutputList("LUMISECTION");
    lumisummaryQuery->addToOutputList("INSTLUMI");
    lumisummaryQuery->addToOutputList("DELIVLUMI");
    lumisummaryQuery->addToOutputList("LIVELUMI");
    lumisummaryQuery->addToOutputList("DEADTIME");
    lumisummaryQuery->setCondition(conditionStr,lumisummaryBindVariables);
    lumisummaryQuery->defineOutput(lumisummaryOutput);
    coral::ICursor& lumisummarycursor=lumisummaryQuery->execute();
    unsigned int rowcounter=0;
    while( lumisummarycursor.next() ){
      const coral::AttributeList& row=lumisummarycursor.currentRow();
      unsigned int lsnum=row["LUMISECTION"].data<unsigned int>();
      float instlumi=row["INSTLUMI"].data<float>();
      float intgdellumi=row["DELIVLUMI"].data<float>();
      float intgreclumi=row["LIVELUMI"].data<float>();
      float deadfraction=row["DEADTIME"].data<float>();
      unsigned short cmsalive=row["CMS_ACTIVE"].data<unsigned short>();
      boost::shared_ptr<DIPLumiSummary> tmpls(new DIPLumiSummary(instlumi,intgdellumi,intgreclumi,deadfraction,cmsalive));
      m_lscache.insert(std::make_pair(lsnum,tmpls));
      ++rowcounter;
    }
    if (rowcounter==0){
      m_isNullRun=true;
      delete lumisummaryQuery;
      return;
    }
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
DIPLumiProducer::clearcache(){
  m_lscache.clear();
  m_cachedrun=0;
}
DIPLumiProducer::~DIPLumiProducer(){}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(DIPLumiProducer);
