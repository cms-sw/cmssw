#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


/** \class HcalRecHitMonitor
  *  
  * $Date: 2008/08/17 15:15:20 $
  * $Revision: 1.18 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  // For now, keep the ZDC stuff separate from the rest of the processEvent code.  Merge it once ZDC is stable
  void processZDC(const ZDCRecHitCollection& zdcHits);
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
    MonitorElement* meRECHIT_T_all;

    MonitorElement* meOCC_MAPthresh_GEO;
    MonitorElement* meRECHIT_Ethresh_tot;
    MonitorElement* meRECHIT_Tthresh_all;
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
 
  MonitorElement* meRECHIT_Ethresh_all;
  MonitorElement* meRECHIT_E_all;
  MonitorElement* meEVT_;

  MonitorElement* hfshort_meRECHIT_E_all;
  MonitorElement* hfshort_meRECHIT_E_low;
  MonitorElement* hfshort_meRECHIT_T_all;

  // ZDC plots
  // TH1F
  MonitorElement* ZDCtanAlpha;
  MonitorElement* ZDCaverageX;
  // TH2F
  MonitorElement* ZDCxplusVSxminus;
  MonitorElement* ZDChadVSem_plus;
  MonitorElement* ZDChadVSem_minus;
  MonitorElement* ZDCenergy_plusVSminus;
  
  // TProfile
  MonitorElement* ZDCenergyVSlayer_plus;
  MonitorElement* ZDCenergyVSlayer_minus;

};

#endif
