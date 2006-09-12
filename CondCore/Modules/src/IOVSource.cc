#include "CondCore/Modules/src/IOVSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "DataFormats/Common/interface/EventID.h"

#include <iostream>
namespace cond{
  //allowed parameters: firstRun, firstTime, lastRun, lastTime, 
  //common paras: connect, catalog, authenticationMethod, timetype
  IOVSource::IOVSource(edm::ParameterSet const& pset,
		       edm::InputSourceDescription const& desc):
    edm::ConfigurableInputSource(pset,desc) {
    m_connect=pset.getParameter<std::string>("connect") ;
    m_catconnect=pset.getUntrackedParameter<std::string>("catalog","");
    m_timeType=pset.getUntrackedParameter<std::string>("timetype","runnumber");
    unsigned int message_level=pset.getUntrackedParameter<unsigned int>("messagelevel",0);
    for(unsigned long long i=1; i<10; i+=2){
      m_iovs.insert(i);//fake iov
    }
    if( m_timeType=="runnumber" ){
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstRun",0);
      m_lastValid=(unsigned long long)pset.getUntrackedParameter<unsigned int>("lastRun",0);
    }else{
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstTime",0);
      m_lastValid=(unsigned long long)pset.getUntrackedParameter<unsigned int>("lastTime",0);
    }
    if(m_firstValid==0){
      m_iovit=m_iovs.begin();
    }else{
      std::set<unsigned long long>::iterator startpos=m_iovs.lower_bound((unsigned long long)m_firstValid);
      m_iovit=startpos;
    }
    if(m_lastValid==0){
      m_iovstop=m_iovs.end();
    }else{
      std::set<unsigned long long>::iterator stoppos=m_iovs.upper_bound((unsigned long long)m_lastValid);
      m_iovstop=stoppos;
    }
  }
  IOVSource::~IOVSource() {
  }
  bool IOVSource::produce( edm::Event & e ) {
    if( (m_iovit==m_iovstop) ){
      return false;
    }
    m_iovit++; 
    return true;
  }
  
  void IOVSource::setRunAndEventInfo(){
    if(m_iovit==m_iovs.end()) return;
    m_currentValid=*m_iovit;
    if( m_timeType=="runnumber" ){
      setRunNumber(m_currentValid);
    }else{
      setTime(m_currentValid);
    }
    setEventNumber(1); 
  }
}
