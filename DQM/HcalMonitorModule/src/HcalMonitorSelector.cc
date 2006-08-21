#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

/*
 * \file HcalMonitorSelector.cc
 * 
 * $Date: 2006/08/16 15:29:57 $
 * $Revision: 1.4 $
 * \author W Fisher
 *
*/

HcalMonitorSelector::HcalMonitorSelector(const edm::ParameterSet& ps){

  m_eventMask = 0;
  m_runNum = -1;
}

HcalMonitorSelector::~HcalMonitorSelector(){

}

void HcalMonitorSelector::processEvent(const edm::Event& e){
  m_eventMask = 0;

  ///Just pass these through for now...
  m_eventMask = DO_HCAL_DIGIMON|DO_HCAL_DFMON|DO_HCAL_RECHITMON;

  edm::Handle<HcalTBTriggerData> triggerD;
  try{
    e.getByType(triggerD);
  }
  catch(...) { 
    m_runNum=-1; 
    //If we don't have the trigger data, just activate everyone!
    m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON|DO_HCAL_LED_CALIBMON; 
    return; 
  }
  const HcalTBTriggerData trigger = *triggerD;
  
  m_runNum = trigger.runNumber();

  // check trigger contents
  if (trigger.wasBeamTrigger()) m_eventMask = m_eventMask|HCAL_BEAM_TRIGGER;
  if (trigger.wasOutSpillPedestalTrigger()) m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;
  if (trigger.wasInSpillPedestalTrigger()) m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;
  if (trigger.wasLEDTrigger()) m_eventMask = m_eventMask|DO_HCAL_LED_CALIBMON;
  if (trigger.wasLaserTrigger()) m_eventMask = m_eventMask|DO_HCAL_LASER_CALIBMON;

  return;
}
