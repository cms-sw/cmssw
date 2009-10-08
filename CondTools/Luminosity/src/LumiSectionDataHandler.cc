#include "CondTools/Luminosity/interface/LumiSectionDataHandler.h"
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
lumi::LumiSectionDataHandler::LumiSectionDataHandler(const edm::ParameterSet& pset):m_name(pset.getParameter<std::string>("lumiRetrieverName")),m_asseed(pset.getParameter<bool>("allowForceFirstSince")){
  m_to_transfer.reserve(300);
  m_datareader=lumi::LumiRetrieverFactory::get()->create( m_name,pset );
}

lumi::LumiSectionDataHandler::~LumiSectionDataHandler()
{
  delete m_datareader;
}

void
lumi::LumiSectionDataHandler::getNewObjects(){
  if(m_asseed){
    if( tagInfo().size==0 ){
      //if it is a new tag, force the *first* since to the begin of time
     m_userTextLog = m_datareader->fill(m_to_transfer,true);
     m_userTextLog += ";firstSince ajusted";
    }else{
      m_userTextLog = m_datareader->fill(m_to_transfer,false);
    }
  }else{
    m_userTextLog = m_datareader->fill(m_to_transfer,false);
  }
}

std::string 
lumi::LumiSectionDataHandler::id() const{
  return m_name;
}


