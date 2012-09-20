#include "DQM/CastorMonitor/interface/CastorMonitorSelector.h"

CastorMonitorSelector::CastorMonitorSelector(const edm::ParameterSet& ps){

  m_eventMask = 0;
  m_triggerMask = 0;
  m_runNum = -1;
}

CastorMonitorSelector::~CastorMonitorSelector(){

}

void CastorMonitorSelector::processEvent(const edm::Event& e){
 
   m_eventMask = 0;
  m_triggerMask = 0;

  ////---- just pass these through for now.
  m_eventMask = DO_CASTOR_RECHITMON;
  m_eventMask = m_eventMask|DO_CASTOR_PED_CALIBMON; 

  return;
}
