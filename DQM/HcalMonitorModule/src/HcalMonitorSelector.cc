#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

/*
 * \file HcalMonitorSelector.cc
 * 
 * $Date: 2005/11/30 22:06:34 $
 * $Revision: 1.1 $
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

  edm::Handle<HcalTBTriggerData> triggerD;
  e.getByType(triggerD);
  const HcalTBTriggerData trigger = *triggerD;
  
  m_runNum = trigger.runNumber();

  // check trigger contents
  if (trigger.wasBeamTrigger()) m_eventMask = m_eventMask|HCAL_BEAM_TRIGGER;
  if (trigger.wasOutSpillPedestalTrigger()) m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;
  if (trigger.wasInSpillPedestalTrigger()) m_eventMask = m_eventMask|DO_HCAL_PED_CALIBMON;
  if (trigger.wasLEDTrigger()) m_eventMask = m_eventMask|DO_HCAL_LED_CALIBMON;
  if (trigger.wasLaserTrigger()) m_eventMask = m_eventMask|DO_HCAL_LASER_CALIBMON;

  ///Just pass these through for now...
  m_eventMask = m_eventMask|DO_HCAL_DIGIMON;
  m_eventMask = m_eventMask|DO_HCAL_DFMON;
  m_eventMask = m_eventMask|DO_HCAL_RECHITMON;

  return;
}
