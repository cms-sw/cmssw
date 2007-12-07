#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"


/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2007/11/28 11:48:44 $
  * $Revision: 1.4 $
  * \author W. Fisher - FNAL
  */
class HcalTrigPrimMonitor: public HcalBaseMonitor {
public:
  HcalTrigPrimMonitor(); 
  ~HcalTrigPrimMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbhedigi,
		    const HODigiCollection& hodigi,
		    const HFDigiCollection& hfdigi,		    
		    const HcalTrigPrimDigiCollection& tpDigis);
  void clearME();
  void reset();

private:  ///Monitoring elements

  int ievt_;
  int etaBins_, phiBins_;

  double occThresh_;
  double etaMax_, etaMin_, phiMax_, phiMin_;

  MonitorElement* meEVT_;
  MonitorElement* tpCount_;
  MonitorElement* tpCountThr_;
  MonitorElement* tpSize_;

  MonitorElement* tpSpectrum_[10];
  MonitorElement* tpSpectrumAll_;
  MonitorElement* tpETSumAll_;
  MonitorElement* tpSOI_ET_;

  MonitorElement* OCC_MAP_SLB;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_MAP_GEO;
  MonitorElement* OCC_MAP_THR;
  MonitorElement* OCC_ELEC_VME;
  MonitorElement* OCC_ELEC_DCC;
  MonitorElement* EN_ETA;
  MonitorElement* EN_PHI;
  MonitorElement* EN_MAP_GEO;
  MonitorElement* EN_ELEC_VME;
  MonitorElement* EN_ELEC_DCC;

};

#endif
