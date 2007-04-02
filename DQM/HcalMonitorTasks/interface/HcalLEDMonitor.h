#ifndef DQM_HCALMONITORTASKS_HCALLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLEDMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

/** \class HcalLEDMonitor
  *  
  * $Date: 2006/09/01 15:39:27 $
  * $Revision: 1.5 $
  * \author W. Fisher - FNAL
  */
class HcalLEDMonitor: public HcalBaseMonitor {
public:
  HcalLEDMonitor(); 
  ~HcalLEDMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

  void done();
  void clearME();


 private: //members vars...

  void perChanHists(int type, const HcalDetId detid, float* vals, 
		    map<HcalDetId, MonitorElement*> &tShape, 
		    map<HcalDetId, MonitorElement*> &tTime);
  map<HcalDetId, MonitorElement*>::iterator _meo;

  bool doPerChannel_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  
  int sigS0_, sigS1_;

  int ievt_, jevt_;
  HcalCalibrations calibs_;


 private: //monitoring elements...
  MonitorElement* meEVT_;

  struct{
    map<HcalDetId,MonitorElement*> shape;
    map<HcalDetId,MonitorElement*> time;

    MonitorElement* shapePED;
    MonitorElement* shapeALL;
    MonitorElement* timeALL;

    MonitorElement* rms_shape;
    MonitorElement* mean_shape;
    MonitorElement* rms_time;
    MonitorElement* mean_time;
    MonitorElement* err_map_geo;
    MonitorElement* err_map_elec;
    /*
    MonitorElement* mean_map_raw;
    MonitorElement* rms_map_raw;
    MonitorElement* mean_map_norm;
    MonitorElement* rms_map_norm;
    MonitorElement* adc_raw;
    MonitorElement* adc_norm;
    */

  } hbHists, heHists, hfHists, hoHists;

  /*
  MonitorElement* mean_map_raw;
  MonitorElement* rms_map_raw;
  MonitorElement* mean_map_norm;
  MonitorElement* rms_map_norm;
  */

};

#endif
