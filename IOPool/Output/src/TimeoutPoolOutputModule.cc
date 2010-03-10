#include "IOPool/Output/interface/TimeoutPoolOutputModule.h"
#include "IOPool/Output/src/RootOutputFile.h"
namespace edm {
  TimeoutPoolOutputModule::TimeoutPoolOutputModule(ParameterSet const& ps):
      PoolOutputModule(ps), 
      m_lastEvent(time(NULL)),
      m_timeout(-1) // we want the first event right away
  {  }

  bool TimeoutPoolOutputModule::shouldWeCloseFile() const {
    if ( PoolOutputModule::shouldWeCloseFile() ) return true;
    time_t now(time(NULL));
    if ( now - m_lastEvent < m_timeout ) return false;
    // next files are needed in 15, 30 and 60 sec
    m_lastEvent = now;
    if (m_timeout == 30) m_timeout = 60;
    if (m_timeout == 15) m_timeout = 30;
    if (m_timeout == -1) m_timeout = 15;
    
    std::cout <<" TimeoutPoolOutputMOdule Closing file "<< currentFileName()<<std::endl;
    return true;
  }
}
