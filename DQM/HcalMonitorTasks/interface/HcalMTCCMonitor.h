#ifndef DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "TH1F.h"


/** \class HcalMTCCmonitor
  *  
  * $Date: 2006/08/24 23:44:59 $
  * $Revision: 1.2 $
  * \author W. Fisher - FNAL
  */
class HcalMTCCMonitor: public HcalBaseMonitor {
public:
  HcalMTCCMonitor(); 
  ~HcalMTCCMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  
  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const LTCDigiCollection& ltc);
  void clearME();

private: 

  int ievt_;
  double occThresh_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

private:  ///Monitoring elements

  MonitorElement* meEVT_;
  MonitorElement* meTrig_;

  struct{
    MonitorElement* DT;
    MonitorElement* CSC;
    MonitorElement* RBC1;
    MonitorElement* RBC2;
    MonitorElement* RBCTB;
    MonitorElement* NA;
    MonitorElement* OCC;
    MonitorElement* E;
    MonitorElement* Etot;
  } hbP, hbM, heP, heM, hoP1, hoM1, hoP2, hoM2; 

};

#endif
