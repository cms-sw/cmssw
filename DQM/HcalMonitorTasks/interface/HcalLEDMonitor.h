#ifndef DQM_HCALMONITORTASKS_HCALLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLEDMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

/** \class HcalLEDMonitor
  *  
  * $Date: 2007/04/25 23:12:16 $
  * $Revision: 1.8 $
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
		    const HcalDbService& cond,
		    const HcalUnpackerReport& report);

  void done();
  void clearME();


 private: //members vars...

  void perChanHists(int type, const HcalDetId detid, float* vals, 
		    map<HcalDetId, MonitorElement*> &tShape, 
		    map<HcalDetId, MonitorElement*> &tTime, 
		    map<HcalDetId, MonitorElement*> &tEnergy);

  void createFEDmap(unsigned int fed);

  map<HcalDetId, MonitorElement*>::iterator _meIter;
  map<unsigned int, MonitorElement*>::iterator _fedIter;

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

  map<unsigned int,MonitorElement*> MEAN_MAP_ENERGY_DCC;
  map<unsigned int,MonitorElement*> RMS_MAP_ENERGY_DCC;
  
  map<unsigned int,MonitorElement*> MEAN_MAP_SHAPE_DCC;
  map<unsigned int,MonitorElement*> RMS_MAP_SHAPE_DCC;

  map<unsigned int,MonitorElement*> MEAN_MAP_TIME_DCC;
  map<unsigned int,MonitorElement*> RMS_MAP_TIME_DCC;

  MonitorElement* FED_UNPACKED;

};

#endif
