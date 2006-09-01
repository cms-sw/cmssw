#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalHotCellMonitor
  *  
  * $Date: 2006/08/24 23:44:59 $
  * $Revision: 1.8 $
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

  MonitorElement* meOCC_MAP_all_GEO;
  MonitorElement* meEN_MAP_all_GEO;
  MonitorElement* meMAX_E_all;
  MonitorElement* meMAX_T_all;
  MonitorElement* meEVT_;

};

#endif
