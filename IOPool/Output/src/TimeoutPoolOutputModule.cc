#include "IOPool/Output/interface/TimeoutPoolOutputModule.h"
#include "IOPool/Output/src/RootOutputFile.h"
namespace edm {

  void TimeoutPoolOutputModule::write(EventPrincipal const& e) {
    eventsWrittenInCurrentFile++;
    PoolOutputModule::write(e);
  }
  
  TimeoutPoolOutputModule::TimeoutPoolOutputModule(ParameterSet const& ps):
      PoolOutputModule(ps), 
      m_lastEvent(time(NULL)),
      eventsWrittenInCurrentFile(0),
      m_timeout(-1) // we want the first event right away
  {  }

  bool TimeoutPoolOutputModule::shouldWeCloseFile() const {
    if ( PoolOutputModule::shouldWeCloseFile() ) {
      eventsWrittenInCurrentFile = 0;
      return true;
    }
    time_t now(time(NULL));
    //    std::cout <<" Events "<< eventsWrittenInCurrentFile<<" time "<< now - m_lastEvent<<std::endl;
    if (eventsWrittenInCurrentFile==0) return false;
    if ( now - m_lastEvent < m_timeout ) return false;
    // next files are needed in 15, 30 and 60 sec
    m_lastEvent = now;
    if (m_timeout == 30) m_timeout = 60;
    if (m_timeout == 15) m_timeout = 30;
    if (m_timeout == -1) m_timeout = 15;
    
    std::cout <<" TimeoutPoolOutputMOdule Closing file "<< currentFileName()<< " with "<< eventsWrittenInCurrentFile  <<" events."<<std::endl;
    eventsWrittenInCurrentFile = 0;
    return true;
  }
}

