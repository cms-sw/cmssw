#ifndef DQM_HCALMONITORTASKS_HCALLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLEDMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/** \class HcalLEDMonitor
  *  
  * $Date: 2008/03/01 00:39:58 $
  * $Revision: 1.13 $
  * \author W. Fisher - FNAL
  */
  static const float LedMonAdc2fc[128]={-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
				     8.5,  9.5, 10.5, 11.5, 12.5, 13.5, 15., 17., 
				     19., 21., 23., 25., 27., 29.5,
				     32.5, 35.5, 38.5, 42., 46., 50., 54.5, 59.5, 
				     64.5, 59.5, 64.5, 69.5, 74.5,
				     79.5, 84.5, 89.5, 94.5, 99.5, 104.5, 109.5, 
				     114.5, 119.5, 124.5, 129.5, 137.,
				     147., 157., 167., 177., 187., 197., 209.5, 224.5, 
				     239.5, 254.5, 272., 292.,
				     312., 334.5, 359.5, 384.5, 359.5, 384.5, 409.5, 
				     434.5, 459.5, 484.5, 509.5,
				     534.5, 559.5, 584.5, 609.5, 634.5, 659.5, 684.5, 709.5, 
				     747., 797., 847.,
				     897.,  947., 997., 1047., 1109.5, 1184.5, 1259.5, 
				     1334.5, 1422., 1522., 1622.,
				     1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5, 
				     2234.5, 2359.5, 2484.5,
				     2609.5, 2734.5, 2859.5, 2984.5, 3109.5, 
				     3234.5, 3359.5, 3484.5, 3609.5, 3797.,
				     4047., 4297., 4547., 4797., 5047., 5297., 
				     5609.5, 5984.5, 6359.5, 6734.5,
				     7172., 7672., 8172., 8734.5, 9359.5, 9984.5};




class HcalLEDMonitor: public HcalBaseMonitor {

public:
  HcalLEDMonitor(); 
  ~HcalLEDMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

  void reset();

  void done();

 private: //members vars...

  // and this is just a very dumb way of doing the adc->fc conversion in the
  // full range (and is the same for all channels and cap-ids)




  void perChanHists(int type, const HcalDetId detid, float* vals, 
		    map<HcalDetId, MonitorElement*> &tShape, 
		    map<HcalDetId, MonitorElement*> &tTime, 
		    map<HcalDetId, MonitorElement*> &tEnergy,
		    string baseFolder);

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


  MonitorElement* HFlumi_ETsum_perwedge;
  MonitorElement* HFlumi_Occupancy_above_thr_r1;
  MonitorElement* HFlumi_Occupancy_between_thrs_r1;
  MonitorElement* HFlumi_Occupancy_below_thr_r1;
  MonitorElement* HFlumi_Occupancy_above_thr_r2;
  MonitorElement* HFlumi_Occupancy_between_thrs_r2;
  MonitorElement* HFlumi_Occupancy_below_thr_r2;


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

};

#endif
