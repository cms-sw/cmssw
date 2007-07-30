#ifndef DQM_HCALMONITORTASKS_HCALLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLEDMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

/** \class HcalLEDMonitor
  *  
  * $Date: 2006/08/24 23:44:59 $
  * $Revision: 1.4 $
  * \author W. Fisher - FNAL
  */
class HcalLEDMonitor: public HcalBaseMonitor {
public:
  HcalLEDMonitor(); 
  ~HcalLEDMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf);

  void done();
  void clearME();

private: 

  void perChanHists(int type, const HcalDetId detid, float* vals, map<HcalDetId, MonitorElement*> &tShape, 
				  map<HcalDetId, MonitorElement*> &tPed, map<HcalDetId, MonitorElement*> &tSig,
				  map<HcalDetId, MonitorElement*> &tTail,  map<HcalDetId, MonitorElement*> &tTime);
  bool m_doPerChannel;
  map<HcalDetId, MonitorElement*>::iterator _meo;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  
  int ievt_, jevt_;
  MonitorElement* meEVT_;

  struct{
    map<HcalDetId,MonitorElement*> shape;
    map<HcalDetId,MonitorElement*> time;
    map<HcalDetId,MonitorElement*> pedRange;
    map<HcalDetId,MonitorElement*> sigRange;
    map<HcalDetId,MonitorElement*> tailRange;

    MonitorElement* shapePED;
    MonitorElement* shapeALL;
    MonitorElement* timeALL;
    MonitorElement* rms_ped;
    MonitorElement* mean_ped;
    MonitorElement* rms_sig;
    MonitorElement* mean_sig;
    MonitorElement* rms_tail;
    MonitorElement* mean_tail;
    MonitorElement* rms_time;
    MonitorElement* mean_time;
    MonitorElement* err_map_geo;
    MonitorElement* err_map_elec;

  } hbHists, hfHists, hoHists;

};

#endif
