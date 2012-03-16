// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      DIPLumiProducer
// 
/**\class DIPLumiProducer DIPLumiProducer.cc RecoLuminosity/LumiProducer/src/DIPLumiProducer.cc
Description: A essource/esproducer for lumi values from DIP via runtime logger DB
*/
// $Id$

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

DIPLumiProducer::DIPLumiProducer(const edm::ParameterSet& ipset):m_pset(ipset),m_cachedrun(0){
  std::cout<<"inDIPLUMI"<<std::endl;
  setWhatProduced(this);
  findingRecord<DIPLumiSummaryRcd>();
}

DIPLumiProducer::ReturnType
DIPLumiProducer::produce(const DIPLumiSummaryRcd&)  
{ 
  std::cout<<"in produce "<<std::endl;
  return m_result;
}

void 
DIPLumiProducer::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, 
				 const edm::IOVSyncValue& iTime, 
				 edm::ValidityInterval& oValidity ) {
  oValidity=edm::ValidityInterval::invalidInterval();
  
  unsigned int currentrun=iTime.eventID().run();
  unsigned int currentls=iTime.luminosityBlockNumber();
  std::cout<<"setIntervalFor run "<<currentrun<<", ls "<<currentls<<std::endl;
  std::cout<<"cached run "<<m_cachedrun<<std::endl;
  if(currentls==0||currentls==4294967295)return;//a fake set
  if(m_cachedrun!=currentrun){
    clearcache();
    m_cachedrun=currentrun;
    fillcache(currentrun,0);//starting ls
    m_result=m_lscache[currentls];//copy construct
  }else{
    if(m_lscache.find(currentls)==m_lscache.end()){//if ls not cached
      std::cout<<"ls not cached, cache it and +50"<<std::endl;
      fillcache(currentrun,currentls);//cache all ls>=currentls for this run
      if(m_lscache.find(currentls)==m_lscache.end()){
	std::cout<<"really no data found "<<std::endl;
	return;
      }
    }else{
      std::cout<<"ls found in the cache "<<std::endl;
    }
    m_result=m_lscache[currentls];//copy construct
  }
  oValidity.setFirst(iTime);
  oValidity.setLast(iTime);
}

void
DIPLumiProducer::fillcache(unsigned int runnumber,unsigned int startlsnum){
  std::cout<<"fillcache cached run: "<<m_cachedrun<<" tofill "<<runnumber<<" , "<<startlsnum<<std::endl;
  if(startlsnum==0){
    for(unsigned int fakels=1;fakels<300;++fakels){
      boost::shared_ptr<DIPLumiSummary> tmpls(new DIPLumiSummary(0.5*fakels,1.5,1.5,0.9,1));
      m_lscache.insert(std::make_pair(fakels,tmpls));
    }
  }else{
    for(unsigned int fakels=startlsnum;fakels<startlsnum+50;++fakels){
      boost::shared_ptr<DIPLumiSummary> tmpls(new DIPLumiSummary(0.5*fakels,1.5,1.5,0.9,1));
      m_lscache.insert(std::make_pair(fakels,tmpls));
    }
  }
}  
void
DIPLumiProducer::clearcache(){
  m_lscache.clear();
  m_cachedrun=0;
}
DIPLumiProducer::~DIPLumiProducer(){}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(DIPLumiProducer);
