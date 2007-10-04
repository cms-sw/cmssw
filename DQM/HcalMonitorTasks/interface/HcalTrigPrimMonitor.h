#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"


/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2007/10/02 22:16:03 $
  * $Revision: 1.1 $
  * \author W. Fisher - FNAL
  */
class HcalTrigPrimMonitor: public HcalBaseMonitor {
public:
  HcalTrigPrimMonitor(); 
  ~HcalTrigPrimMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HcalTrigPrimDigiCollection& tpDigis);
  void clearME();
  void reset();

private:  ///Monitoring elements

  int ievt_;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

  MonitorElement* meEVT_;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_MAP_GEO;
  MonitorElement* OCC_ELEC_VME;
  MonitorElement* OCC_ELEC_DCC;

};

#endif
