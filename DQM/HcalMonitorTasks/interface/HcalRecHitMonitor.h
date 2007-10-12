#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalRecHitMonitor
  *  
  * $Date: 2006/09/01 15:39:27 $
  * $Revision: 1.9 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  void clearME();

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
 
  MonitorElement* meRECHIT_E_all;
  MonitorElement* meEVT_;


};

#endif
