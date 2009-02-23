#include "CondTools/RunInfo/interface/LuminosityInfoHandler.h"
#include "CondTools/RunInfo/interface/LumiReaderBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondTools/RunInfo/interface/LumiReaderFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
//#include <iostream>
lumi::LuminosityInfoHandler::LuminosityInfoHandler(const edm::ParameterSet& pset):m_name(pset.getParameter<std::string>("lumiReaderName")),m_startRun(0),m_numberOfRuns(0),m_lumiversionNumber(0){
  m_to_transfer.reserve(100);
  m_startRun=pset.getUntrackedParameter<int>("startRun");
  m_numberOfRuns=pset.getUntrackedParameter<int>("numberOfRuns");
  m_lumiversionNumber=(short)pset.getParameter<int>("lumiVersionNumber");
  m_datareader=lumi::LumiReaderFactory::get()->create(m_name,pset);
}

lumi::LuminosityInfoHandler::~LuminosityInfoHandler()
{
  delete m_datareader;
}

void
lumi::LuminosityInfoHandler::getNewObjects(){
  //edm::LuminosityBlockID lumiblockID(m_startRun,1);  
  std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> > result;
  m_datareader->fill(m_startRun,m_numberOfRuns,result,m_lumiversionNumber);
  std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >::const_iterator iBeg=result.begin();
  std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >::const_iterator iEnd=result.end();
  std::copy(result.begin(),result.end(),std::back_inserter(m_to_transfer));
  /*for(std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >::const_iterator it=iBeg; it!=iEnd;++it){
    m_to_transfer.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(it->first,it->second));
    //std::copy(result.begin(),result.end(),std::back_inserter(m_to_transfer));
    }
  */
}

std::string 
lumi::LuminosityInfoHandler::id() const{
  return m_name;
}


