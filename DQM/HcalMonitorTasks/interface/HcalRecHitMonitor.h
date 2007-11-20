#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"


/** \class HcalRecHitMonitor
  *  
  * $Date: 2007/10/23 14:17:17 $
  * $Revision: 1.13 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  void reset();

private:  ///Monitoring elements

  bool doPerChannel_;
  float occThresh_;
  int ievt_;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

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
 
  MonitorElement* meRECHIT_E_all;
  MonitorElement* meEVT_;


};

#endif
