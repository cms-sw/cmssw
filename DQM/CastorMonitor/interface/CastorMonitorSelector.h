#ifndef CastorMonitorSelector_H
#define CastorMonitorSelector_H


#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include <iostream>
#include <vector>
#include <string>


static const int CASTOR_BEAM_TRIGGER       = 0x0001;
static const int DO_CASTOR_RECHITMON       = 0x0008;
static const int DO_CASTOR_LED_CALIBMON    = 0x0010;
static const int DO_CASTOR_PED_CALIBMON    = 0x0040;


class CastorMonitorSelector{

public:

/// Constructor
CastorMonitorSelector() {};
CastorMonitorSelector(const edm::ParameterSet& ps);

/// Destructor
~CastorMonitorSelector();

  inline unsigned int getEventMask() const { return m_eventMask; }
  inline unsigned int getTriggerMask() const { return m_triggerMask; }
  inline int getRunNumber() const { return m_runNum; }
  void processEvent(const edm::Event& e);
  
protected:

private:

  unsigned int m_eventMask;
  unsigned int m_triggerMask;
  int m_runNum;
};

#endif
