#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "DummySource.h"
#include <cmath>
//#include <iostream>

lumi::DummySource::DummySource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
  m_lumiversion=pset.getParameter<std::string>("lumiVersion");
  m_runnumber=pset.getParameter<int>("runNumber");
  
}

const std::string
lumi::DummySource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result, bool allowForceFirstSince ){
  //std::cout<<"lumiversion "<<m_lumiversion<<std::endl;
  std::vector<lumi::TriggerInfo> triginfo;
  triginfo.reserve(192);
  for(size_t i=0;i<192;++i){
    lumi::TriggerInfo trgbit("fake",123,12893,1);
    triginfo.push_back(trgbit);
  }
  std::vector<lumi::HLTInfo> hltinfo;
  hltinfo.reserve(200);
  for(size_t i=0;i<200;++i){
    lumi::HLTInfo hltperpath("fake",2829345,1234,1);
    hltinfo.push_back(hltperpath);
  }
  size_t runnumber=0;
  size_t lumisecnumber=0;
  for(size_t j=1; j<2; ++j){
    if(allowForceFirstSince && j==1){  //if allowForceFirstSince and this is the head of the iov, then set the head to the begin of time
      runnumber=1;
      lumisecnumber=1;
    }else{
      runnumber=m_runnumber;
      lumisecnumber=j;
    }
    edm::LuminosityBlockID lu(runnumber,lumisecnumber);
    cond::Time_t current=(cond::Time_t)(lu.value());
    lumi::LumiSectionData* l=new lumi::LumiSectionData;
    l->setLumiVersion(m_lumiversion);
    l->setLumiSectionId(j);
    l->setLumiAverage(1.1);
    l->setLumiQuality(1);
    l->setDeadFraction(0.2);
    l->setLumiError(0.5);
    l->setStartOrbit((j-1)*(unsigned long long)pow(2,20)+1); //assuming count from 1
    std::vector<lumi::BunchCrossingInfo> bxinfo;
    bxinfo.reserve(3564);
    for(int bxidx=1;bxidx<=3564;++bxidx){
      bxinfo.push_back(lumi::BunchCrossingInfo(bxidx,2.1,0.6,3));
    }
    l->setBunchCrossingData(bxinfo,lumi::ET);
    l->setBunchCrossingData(bxinfo,lumi::OCCD1);
    l->setBunchCrossingData(bxinfo,lumi::OCCD2);
    l->setQualityFlag(1);
    l->setTriggerData(triginfo);
    l->setHLTData(hltinfo);
    //std::cout<<"current "<<current<<std::endl;
    result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,current));
  }
  return std::string("DummySource");
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::DummySource,"dummysource");
 
