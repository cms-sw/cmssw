#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalHotCellMonitor
  *  
  * $Date: 2006/09/01 15:39:27 $
  * $Revision: 1.1 $
  * \author W. Fisher - FNAL
  */
class HcalHotCellMonitor: public HcalBaseMonitor {
public:
  HcalHotCellMonitor(); 
  ~HcalHotCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  void clearME();

private:  ///Monitoring elements

  int ievt_;
  double occThresh_;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

  struct{
    MonitorElement* meOCC_MAP_GEO;
    MonitorElement* meEN_MAP_GEO;
    MonitorElement* meMAX_E;
    MonitorElement* meMAX_T;
    MonitorElement* meMAX_ID;
  }hbHists,heHists,hfHists,hoHists;

  MonitorElement* meOCC_MAP_L1;
  MonitorElement* meEN_MAP_L1;
  MonitorElement* meOCC_MAP_L2;
  MonitorElement* meEN_MAP_L2;
  MonitorElement* meOCC_MAP_L3;
  MonitorElement* meEN_MAP_L3;
  MonitorElement* meOCC_MAP_L4;
  MonitorElement* meEN_MAP_L4;

  MonitorElement* meOCC_MAP_all;
  MonitorElement* meEN_MAP_all;

  MonitorElement* meMAX_E_all;
  MonitorElement* meMAX_T_all;
  MonitorElement* meEVT_;

};

#endif
