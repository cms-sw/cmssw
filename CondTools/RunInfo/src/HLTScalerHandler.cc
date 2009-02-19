#include "CondTools/RunInfo/interface/HLTScalerHandler.h"
#include "CondTools/RunInfo/interface/HLTScalerReaderBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/HLTScalerReaderFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
//#include <iostream>
lumi::HLTScalerHandler::HLTScalerHandler(const edm::ParameterSet& pset):m_name(pset.getParameter<std::string>("hltscalerReaderName")),m_startRun(0),m_numberOfRuns(0){
  m_to_transfer.reserve(100);
  m_startRun=pset.getUntrackedParameter<int>("startRun");
  m_numberOfRuns=pset.getUntrackedParameter<int>("numberOfRuns");
  m_datareader=lumi::HLTScalerReaderFactory::get()->create(m_name,pset);
}

lumi::HLTScalerHandler::~HLTScalerHandler()
{
  delete m_datareader;
}

void
lumi::HLTScalerHandler::getNewObjects(){
  //edm::LuminosityBlockID lumiblockID(m_startRun,1);  
  std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> > result;
  m_datareader->fill(m_startRun,m_numberOfRuns,result);
  std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >::const_iterator iBeg=result.begin();
  std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >::const_iterator iEnd=result.end();
  std::copy(result.begin(),result.end(),std::back_inserter(m_to_transfer));
}

std::string 
lumi::HLTScalerHandler::id() const{
  return m_name;
}


