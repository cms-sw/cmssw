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

  edm::Handle<HcalTBTriggerData> triggerD; ////-- where is CastorTBTriggerData ????
  e.getByType(triggerD);
  if (!triggerD.isValid()) {
    m_runNum=-1; 
    ////---- If there is no trigger data, just activate everything
    //m_eventMask = m_eventMask|DO_CASTOR_PED_CALIBMON|DO_CASTOR_LED_CALIBMON|DO_CASTOR_LASER_CALIBMON;
    m_eventMask = m_eventMask|DO_CASTOR_PED_CALIBMON;
    return; 
  }
  const HcalTBTriggerData trigger = *triggerD;
  
  m_runNum = trigger.runNumber();

  ////--- check trigger contents
  //if (trigger.wasBeamTrigger())             { m_triggerMask |= 0x01; m_eventMask = m_eventMask|CASTOR_BEAM_TRIGGER;}
  if (trigger.wasOutSpillPedestalTrigger()) { m_triggerMask |= 0x02; m_eventMask = m_eventMask|DO_CASTOR_PED_CALIBMON;}
  if (trigger.wasInSpillPedestalTrigger())  { m_triggerMask |= 0x04; m_eventMask = m_eventMask|DO_CASTOR_PED_CALIBMON;}
  // if (trigger.wasLEDTrigger())              { m_triggerMask |= 0x08; m_eventMask = m_eventMask|DO_CASTOR_LED_CALIBMON;}
  // if (trigger.wasLaserTrigger())            { m_triggerMask |= 0x10; m_eventMask = m_eventMask|DO_CASTOR_LASER_CALIBMON;}

  if(m_eventMask&DO_CASTOR_PED_CALIBMON) m_eventMask = m_eventMask^DO_CASTOR_RECHITMON;

  return;

}
