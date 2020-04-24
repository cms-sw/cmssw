#include "IOPool/Output/interface/TimeoutPoolOutputModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  void TimeoutPoolOutputModule::write(EventForOutput const& e) {
    eventsWrittenInCurrentFile++;
    PoolOutputModule::write(e);
  }
  
  TimeoutPoolOutputModule::TimeoutPoolOutputModule(ParameterSet const& ps):
      edm::one::OutputModuleBase::OutputModuleBase(ps),
      PoolOutputModule(ps), 
      m_lastEvent(time(nullptr)),
      eventsWrittenInCurrentFile(0),
      m_timeout(-1) // we want the first event right away
  {  }

  void
  TimeoutPoolOutputModule::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    PoolOutputModule::fillDescription(desc);
    descriptions.add("TimeoutPoolOutputModule", desc);
  }

  bool TimeoutPoolOutputModule::shouldWeCloseFile() const {
    time_t now(time(nullptr));
    if ( PoolOutputModule::shouldWeCloseFile() ) {
      edm::LogVerbatim("TimeoutPoolOutputModule")  <<" Closing file "<< currentFileName()<< " with "<< eventsWrittenInCurrentFile  <<" events.";
      eventsWrittenInCurrentFile = 0;
      m_lastEvent = now;
      return true;
    }
    //    std::cout <<" Events "<< eventsWrittenInCurrentFile<<" time "<< now - m_lastEvent<<std::endl;
    if (eventsWrittenInCurrentFile==0) return false;
    if ( now - m_lastEvent < m_timeout ) return false;
    // next files are needed in 15, 30 and 60 sec
    m_lastEvent = now;
    if (m_timeout == 30) m_timeout = 60;
    if (m_timeout == 15) m_timeout = 30;
    if (m_timeout == -1) m_timeout = 15;
    
    edm::LogVerbatim("TimeoutPoolOutputModule")  <<" Closing file "<< currentFileName()<< " with "<< eventsWrittenInCurrentFile  <<" events.";
    eventsWrittenInCurrentFile = 0;
    return true;
  }
}

