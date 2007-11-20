#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

/** \class HcalDigiMonitor
  *  
  * $Date: 2007/10/04 21:03:13 $
  * $Revision: 1.14 $
  * \author W. Fisher - FNAL
  */
class HcalDigiMonitor: public HcalBaseMonitor {
public:
  HcalDigiMonitor(); 
  ~HcalDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

  void reset();

private:  ///Methods

  void fillErrors(const HBHEDataFrame& hb);
  void fillErrors(const HODataFrame& ho);
  void fillErrors(const HFDataFrame& hf);

  int ievt_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  bool doPerChannel_;
  int occThresh_;
  HcalCalibrations calibs_;

private:  ///Monitoring elements
  MonitorElement* meEVT_;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_L1;
  MonitorElement* OCC_L2;
  MonitorElement* OCC_L3;
  MonitorElement* OCC_L4;
  MonitorElement* OCC_ELEC_VME;
  MonitorElement* OCC_ELEC_FIB;
  MonitorElement* OCC_ELEC_DCC;
  MonitorElement* ERR_MAP_GEO;
  MonitorElement* ERR_MAP_VME;
  MonitorElement* ERR_MAP_FIB;
  MonitorElement* ERR_MAP_DCC;

  struct{
    MonitorElement* DIGI_NUM;
    MonitorElement* DIGI_SIZE;
    MonitorElement* DIGI_PRESAMPLE;
    MonitorElement* QIE_CAPID;
    MonitorElement* QIE_ADC;
    MonitorElement* QIE_DV;
    MonitorElement* ERR_MAP_GEO;
    MonitorElement* ERR_MAP_VME;
    MonitorElement* ERR_MAP_FIB;
    MonitorElement* ERR_MAP_DCC;
    MonitorElement* OCC_MAP_GEO1;
    MonitorElement* OCC_MAP_GEO2;
    MonitorElement* OCC_MAP_GEO3;
    MonitorElement* OCC_MAP_GEO4;
    MonitorElement* OCC_ETA;
    MonitorElement* OCC_PHI;
    MonitorElement* OCC_MAP_VME;
    MonitorElement* OCC_MAP_FIB;
    MonitorElement* OCC_MAP_DCC;
    MonitorElement* SHAPE_tot;
    MonitorElement* SHAPE_THR_tot;
    std::map<HcalDetId, MonitorElement*> SHAPE;
  } hbHists, heHists, hfHists, hoHists;

};

#endif
