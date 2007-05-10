#ifndef DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALMTCCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
//#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "TH1F.h"


/** \class HcalMtccmonitor2
  *  
  * $Date: 2006/10/23 19:31:56 $
  * $Revision: 1.5 $
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
  double dumpEtaLo_, dumpEtaHi_;
  double dumpPhiLo_, dumpPhiHi_;

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
