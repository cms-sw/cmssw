#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

/*
 * \file HcalMonitorSelector.cc
 * 
 * $Date: 2006/09/28 22:17:54 $
 * $Revision: 1.7 $
 * \author W Fisher
 *
*/

HcalMonitorSelector::HcalMonitorSelector(const edm::ParameterSet& ps){

  m_eventMask = 0;
  m_triggerMask = 0;
  m_runNum = -1;
}

HcalMonitorSelector::~HcalMonitorSelector(){

}

void HcalMonitorSelector::processEvent(const edm::Event& e){
  m_eventMask = 0;
  m_triggerMask = 0;

  ///Just pass these through for now...
  m_eventMask = DO_HCAL_DIGIMON|DO_HCAL_DFMON|DO_HCAL_RECHITMON;
  m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON|DO_HCAL_LED_CALIBMON; 

  return;

  edm::Handle<HcalTBTriggerData> triggerD;
  try{
    e.getByType(triggerD);
  }
  catch(exception& ex) { 
    m_runNum=-1; 
    //If we don't have the trigger data, just activate everyone!
    m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON|DO_HCAL_LED_CALIBMON; 
    return; 
  }
  const HcalTBTriggerData trigger = *triggerD;
  
  m_runNum = trigger.runNumber();

  // check trigger contents
  if (trigger.wasBeamTrigger())             { m_triggerMask |= 0x01; m_eventMask = m_eventMask|HCAL_BEAM_TRIGGER;}
  if (trigger.wasOutSpillPedestalTrigger()) { m_triggerMask |= 0x02; m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;}
  if (trigger.wasInSpillPedestalTrigger())  { m_triggerMask |= 0x04; m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;}
  if (trigger.wasLEDTrigger())              { m_triggerMask |= 0x08; m_eventMask = m_eventMask|DO_HCAL_LED_CALIBMON;}
  if (trigger.wasLaserTrigger())            { m_triggerMask |= 0x10; m_eventMask = m_eventMask|DO_HCAL_LASER_CALIBMON;}

  if(m_eventMask&DO_HCAL_PED_CALIBMON) m_eventMask = m_eventMask^DO_HCAL_RECHITMON;

  return;
}
