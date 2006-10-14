#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalRecHitMonitor
  *  
  * $Date: 2006/04/04 19:27:03 $
  * $Revision: 1.4 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);


private:  ///Monitoring elements

  bool doPerChannel_;
  float occThresh_;
  int ievt_;

  struct{
    MonitorElement* meOCC_MAP_GEO;
    MonitorElement* meRECHIT_E_all;
    MonitorElement* meRECHIT_E_tot;
    MonitorElement* meRECHIT_T_tot;
    std::map<HcalDetId, MonitorElement*> meRECHIT_E, meRECHIT_T;  // complicated per-channel histogram setup
  }hbHists,hfHists,hoHists;

  MonitorElement* meOCC_MAP_all_GEO;
  MonitorElement* meRECHIT_E_all;
  MonitorElement* meEVT_;


};

#endif
