#ifndef DQM_CASTORMONITOR_CASTORRECHITMONITOR_H
#define DQM_CASTORMONITOR_CASTORRECHITMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

class CastorRecHitMonitor: public CastorBaseMonitor {
public:
  CastorRecHitMonitor(); 
  ~CastorRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorRecHitCollection& castorHits);
  void reset();

private:  

  bool doPerChannel_;
  // float occThresh_;
  int ievt_;

  ////---- define Monitoring elements
  struct{
    MonitorElement* meRECHIT_E_all     ; //-- energy of all hits 
    MonitorElement* meRECHIT_T_all     ; //-- time of all hits
    MonitorElement* meRECHIT_MAP_CHAN_E; //-- energy vs channel plot

    std::map<HcalCastorDetId, MonitorElement*> meRECHIT_E, meRECHIT_T;  //-- complicated per-channel histogram setup
  } castorHists ;

  MonitorElement* meEVT_;

};

#endif
