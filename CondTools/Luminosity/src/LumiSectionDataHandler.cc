#include "CondTools/Luminosity/interface/LumiSectionDataHandler.h"
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  float deliveredLumi=0.0;
  float recordedLumi=0.0;
  std::string lumiversion;
  size_t nlumi=m_to_transfer.size();
  typedef std::vector<std::pair<lumi::LumiSectionData*,unsigned long long> >::const_iterator lumiIt;
  lumiIt lumiBeg=m_to_transfer.begin();
  lumiIt lumiEnd=m_to_transfer.end();
  for(lumiIt it=lumiBeg;it!=lumiEnd;++it){
    if(it==lumiBeg) lumiversion=it->first->lumiVersion();
    deliveredLumi += it->first->lumiAverage()*93.2;
    recordedLumi += it->first->lumiAverage()*(1-it->first->deadFraction())*93.2;
  }
  edm::LogInfo("LumiReport")<<"Data Source : "<<m_userTextLog<<"\n"
			    <<"Total LS : "<<nlumi
			    <<"\t"<<"LHC Delivered : "<<deliveredLumi
			    <<"\t"<<"CMS Recorded : "<<recordedLumi
			    <<"\t"<<"Version : "<<lumiversion;
}

std::string 
lumi::LumiSectionDataHandler::id() const{
  return m_name;
}


