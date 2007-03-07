#include "CondCore/Modules/src/EmptyIOVSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "CondCore/DBCommon/interface/Exception.h"

namespace cond{
  //allowed parameters: firstRun, firstTime, lastRun, lastTime, 
  //common paras: timetype,interval
  EmptyIOVSource::EmptyIOVSource(edm::ParameterSet const& pset,
				 edm::InputSourceDescription const& desc):
    edm::ConfigurableInputSource(pset,desc),
    m_timeType(pset.getParameter<std::string>("timetype")),
    m_interval((cond::Time_t)pset.getParameter<unsigned int>("interval")){
    unsigned int lastValid;
    if( m_timeType=="runnumber" ){
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstRun",1);
      lastValid=pset.getUntrackedParameter<unsigned int>("lastRun",0);
      if(lastValid==0){
	m_lastValid=edm::IOVSyncValue::endOfTime().eventID().run();
      }else{
	m_lastValid=lastValid;
      }
    }else{
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstTime",0);
      lastValid=(cond::Time_t)pset.getUntrackedParameter<unsigned int>("lastTime",0);
      if(lastValid==0){
	m_lastValid=edm::IOVSyncValue::endOfTime().time().value();
      }else{
	m_lastValid=lastValid;
      }
    }
    for(cond::Time_t i=(cond::Time_t)m_firstValid; i<=m_lastValid; i+=(cond::Time_t)m_interval){
      m_iovs.insert(i);
    }
    if(m_firstValid==0){
      m_iovit=m_iovs.begin();
    }else{
      std::set<cond::Time_t>::iterator startpos=m_iovs.lower_bound((cond::Time_t)m_firstValid);
      m_iovit=startpos;
    }
    if(m_lastValid==0){
      m_iovstop=m_iovs.end();
    }else{
      std::set<cond::Time_t>::iterator stoppos=m_iovs.upper_bound((cond::Time_t)m_lastValid);
      m_iovstop=stoppos;
    }
  }
  EmptyIOVSource::~EmptyIOVSource() {
  }
  bool EmptyIOVSource::produce( edm::Event & e ) {
    if( (m_iovit==m_iovstop) ){
      return false;
    }
    m_iovit++; 
    return true;
  }  
  void EmptyIOVSource::setRunAndEventInfo(){
    if(m_iovit==m_iovs.end()) return;
    m_currentValid=*m_iovit;
    if( m_timeType=="runnumber" ){
      setRunNumber(m_currentValid);
    }else{
      setTime(m_currentValid);
    }
    setEventNumber(1); 
  }
}//ns cond
