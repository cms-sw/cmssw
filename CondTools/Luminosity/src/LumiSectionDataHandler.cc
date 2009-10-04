#include "CondTools/Luminosity/interface/LumiSectionDataHandler.h"
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
//#include <iostream>
lumi::LumiSectionDataHandler::LumiSectionDataHandler(const edm::ParameterSet& pset):m_name(pset.getParameter<std::string>("lumiRetrieverName")),m_runnumber(1),m_lumiversionnumber(1){
  m_to_transfer.reserve(100);
  m_runnumber=pset.getParameter<int>("RunNumber");
  m_lumiversionnumber=(short)pset.getParameter<int>("lumiVersionNumber");
  m_datareader=lumi::LumiRetrieverFactory::get()->create(m_name,pset);
}

lumi::LumiSectionDataHandler::~LumiSectionDataHandler()
{
  delete m_datareader;
}

void
lumi::LumiSectionDataHandler::getNewObjects(){
  std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> > result;
  m_datareader->fill(m_runnumber,result,m_lumiversionnumber);
  std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >::const_iterator iBeg=result.begin();
  std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >::const_iterator iEnd=result.end();
  std::copy(result.begin(),result.end(),std::back_inserter(m_to_transfer));
}

std::string 
lumi::LumiSectionDataHandler::id() const{
  return m_name;
}


