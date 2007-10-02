#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"


/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2007/04/02 13:19:38 $
  * $Revision: 1.11 $
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

private:  ///Monitoring elements

  int ievt_;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

  MonitorElement* meEVT_;
  /*
  struct{
    MonitorElement* meOCC_MAP_GEO;
    MonitorElement* meRECHIT_E_all;
    MonitorElement* meRECHIT_E_low;
    MonitorElement* meRECHIT_E_tot;
    MonitorElement* meRECHIT_T_tot;
    std::map<HcalDetId, MonitorElement*> meRECHIT_E, meRECHIT_T;  // complicated per-channel histogram setup
  }hbHists,heHists, hfHists,hoHists;

  MonitorElement* meOCC_MAP_L1;
  MonitorElement* meOCC_MAP_L1_E;
  MonitorElement* meOCC_MAP_L2;
  MonitorElement* meOCC_MAP_L2_E;
  MonitorElement* meOCC_MAP_L3;
  MonitorElement* meOCC_MAP_L3_E;
  MonitorElement* meOCC_MAP_L4;
  MonitorElement* meOCC_MAP_L4_E;

  MonitorElement* meOCC_MAP_ETA;
  MonitorElement* meOCC_MAP_PHI;
  MonitorElement* meOCC_MAP_ETA_E;
  MonitorElement* meOCC_MAP_PHI_E;
  */

};

#endif
