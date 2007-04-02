#ifndef DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"


/** \class HcalMtccmonitor2
  *  
  * $Date: 2007/03/21 23:01:29 $
  * $Revision: 1.7 $
  * \author W. Fisher - FNAL
  */
class HcalMTCCMonitor: public HcalBaseMonitor {
 public:
  HcalMTCCMonitor(); 
  ~HcalMTCCMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const LTCDigiCollection& ltc,
		    const HcalDbService& cond);
  
  void clearME();

private: 

  void dumpDigi(const HBHEDigiCollection& hbhe, const HODigiCollection& ho, const HcalDbService& cond);

  int ievt_;
  double occThresh_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  const HcalQIEShape* shape_;
  HcalCalibrations calibs_;

  double dumpThresh_;
  int dumpEtaLo_, dumpEtaHi_;
  int dumpPhiLo_, dumpPhiHi_;

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
    MonitorElement* GLTRIG;
    MonitorElement* OCC;
    MonitorElement* E;
    MonitorElement* PEDS;    
  } hbC, heC, hoP1, hoM1, hoP2, hoM2, hoC; 

};

#endif
