#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "NoDataSource.h"
//#include <iostream>

lumi::NoDataSource::NoDataSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){
}

const std::string
lumi::NoDataSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result, bool allowForceFirstSince ){
  //edm::LuminosityBlockNumber_t blockid=edm::LuminosityBlockID::maxLuminosityBlockNumber();
  //edm::RunNumber_t runnum=edm::RunID::maxRunNumber();
  edm::LuminosityBlockID lu(1,1);
  lumi::LumiSectionData* l=new lumi::LumiSectionData;
  l->setLumiNull();
  result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,(cond::Time_t)lu.value()));  
  return std::string("NoDataSource");
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::NoDataSource,"nodatasource");

