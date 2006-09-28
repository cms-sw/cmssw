#ifndef DQM_HCALMONITORTASKS_HCALMTCCMONITORTWO_H
#define DQM_HCALMONITORTASKS_HCALMTCCMONITORTWO_H

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
  * $Date: 2006/08/25 16:07:01 $
  * $Revision: 1.3 $
  * \author W. Fisher - FNAL
  */
class HcalMTCCMonitor2: public HcalBaseMonitor {
 public:
  HcalMTCCMonitor2(); 
  ~HcalMTCCMonitor2(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const LTCDigiCollection& ltc,
		    const HcalDbService& cond);

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
    MonitorElement* GLTRIG;
    MonitorElement* OCC;
    MonitorElement* E;
    MonitorElement* PEDS;    
  } hbP, hbM, heP, heM, hoP1, hoM1, hoP2, hoM2; 

};

#endif
