#ifndef DQM_HCALMONITORTASKS_HCALLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLEDMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

/** \class HcalLEDMonitor
  *  
  * $Date: 2007/04/21 19:45:08 $
  * $Revision: 1.7 $
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
		    map<HcalDetId, MonitorElement*> &tTime, 
		    map<HcalDetId, MonitorElement*> &tEnergy);

  map<HcalDetId, MonitorElement*>::iterator _meo;

  bool doPerChannel_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  
  int sigS0_, sigS1_;
  float adcThresh_;

  int ievt_, jevt_;
  HcalCalibrations calibs_;


 private: //monitoring elements...
  MonitorElement* meEVT_;

  struct{
    map<HcalDetId,MonitorElement*> shape;
    map<HcalDetId,MonitorElement*> time;
    map<HcalDetId,MonitorElement*> energy;

    MonitorElement* shapePED;
    MonitorElement* shapeALL;
    MonitorElement* timeALL;
    MonitorElement* energyALL;

    MonitorElement* rms_shape;
    MonitorElement* mean_shape;

    MonitorElement* rms_time;
    MonitorElement* mean_time;

    MonitorElement* rms_energy;
    MonitorElement* mean_energy;

    MonitorElement* err_map_geo;
    MonitorElement* err_map_elec;


  } hbHists, heHists, hfHists, hoHists;

  /*  
  MonitorElement* mean_map_ADCraw;
  MonitorElement* rms_map_ADCraw;
  MonitorElement* mean_map_ADCnorm;
  MonitorElement* rms_map_ADCnorm;
  */

  MonitorElement* MEAN_MAP_TIME_L1;
  MonitorElement*  RMS_MAP_TIME_L1;

  MonitorElement* MEAN_MAP_TIME_L2;
  MonitorElement*  RMS_MAP_TIME_L2;

  MonitorElement* MEAN_MAP_TIME_L3;
  MonitorElement*  RMS_MAP_TIME_L3;

  MonitorElement* MEAN_MAP_TIME_L4;
  MonitorElement*  RMS_MAP_TIME_L4;

  MonitorElement* MEAN_MAP_SHAPE_L1;
  MonitorElement*  RMS_MAP_SHAPE_L1;

  MonitorElement* MEAN_MAP_SHAPE_L2;
  MonitorElement*  RMS_MAP_SHAPE_L2;

  MonitorElement* MEAN_MAP_SHAPE_L3;
  MonitorElement*  RMS_MAP_SHAPE_L3;

  MonitorElement* MEAN_MAP_SHAPE_L4;
  MonitorElement*  RMS_MAP_SHAPE_L4;

  MonitorElement* MEAN_MAP_ENERGY_L1;
  MonitorElement*  RMS_MAP_ENERGY_L1;

  MonitorElement* MEAN_MAP_ENERGY_L2;
  MonitorElement*  RMS_MAP_ENERGY_L2;

  MonitorElement* MEAN_MAP_ENERGY_L3;
  MonitorElement*  RMS_MAP_ENERGY_L3;

  MonitorElement* MEAN_MAP_ENERGY_L4;
  MonitorElement*  RMS_MAP_ENERGY_L4;


};

#endif
