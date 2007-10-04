#ifndef DQM_HCALMONITORTASKS_HCALCOMMISSIONINGMONITOR_H
#define DQM_HCALMONITORTASKS_HCALCOMMISSIONINGMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"


/** \class HcalCommisioningMonitor
  *  
  * $Date: 2007/04/02 13:23:14 $
  * $Revision: 1.1 $
  * \author W. Fisher - FNAL
  */
class HcalCommissioningMonitor: public HcalBaseMonitor {
 public:
  HcalCommissioningMonitor(); 
  ~HcalCommissioningMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits,
		    const HFRecHitCollection& hfHits,
		    const LTCDigiCollection& ltc,
		    const HcalDbService& cond);
  
  void clearME();
  void reset();

 private: 
  int ievt_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

 private:  ///Monitoring elements

  MonitorElement* meEVT_;

};

#endif
