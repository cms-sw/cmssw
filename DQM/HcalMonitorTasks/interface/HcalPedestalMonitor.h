#ifndef DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H
#define DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

/** \class HcalPedestalMonitor
  *  
  * $Date: 2007/10/04 21:03:13 $
  * $Revision: 1.11 $
  * \author W. Fisher - FNAL
  */
class HcalPedestalMonitor: public HcalBaseMonitor {
public:
  HcalPedestalMonitor(); 
  ~HcalPedestalMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);
  void done();
  void clearME();
  void reset();

private: 
  void perChanHists(int id, vector<HcalDetId> detID, vector<int> capID, vector<float> peds,
		    map<HcalDetId, map<int, MonitorElement*> > &toolP, 
		    map<HcalDetId, map<int, MonitorElement*> > &toolS);

  bool doPerChannel_;
  bool doFCpeds_;
  map<HcalDetId, map<int,MonitorElement*> >::iterator meo_;
  vector<HcalDetId> detID_;
  vector<int> capID_;
  vector<float> pedVals_;

  string outputFile_;
  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;
  HcalCalibrations calibs_;

  MonitorElement* meEVT_;
  int ievt_;
  
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  map<HcalDetId,bool> REG;

  MonitorElement* MEAN_MAP_L1;
  MonitorElement*  RMS_MAP_L1;

  MonitorElement* MEAN_MAP_L2;
  MonitorElement*  RMS_MAP_L2;

  MonitorElement* MEAN_MAP_L3;
  MonitorElement*  RMS_MAP_L3;

  MonitorElement* MEAN_MAP_L4;
  MonitorElement*  RMS_MAP_L4;

  MonitorElement* MEAN_MAP_CR;
  MonitorElement*  RMS_MAP_CR;

  MonitorElement* MEAN_MAP_FIB;
  MonitorElement*  RMS_MAP_FIB;

  MonitorElement* MEAN_MAP_SP;
  MonitorElement*  RMS_MAP_SP;

  MonitorElement* PEDESTAL_REFS;
  MonitorElement* WIDTH_REFS;

  struct{
    map<HcalDetId,map<int, MonitorElement*> > PEDVALS;
    map<HcalDetId,map<int, MonitorElement*> > SUBVALS;
    MonitorElement* ALLPEDS;
    MonitorElement* PEDRMS;
    MonitorElement* PEDMEAN;    

    MonitorElement* SUBMEAN;    
    MonitorElement* NSIGMA;    

    MonitorElement* CAPIDRMS;
    MonitorElement* CAPIDMEAN;    

    MonitorElement* QIERMS;
    MonitorElement* QIEMEAN;    

    MonitorElement* ERRGEO;
    MonitorElement* ERRELEC;    

    MonitorElement* PEDESTAL_REFS;
    MonitorElement* WIDTH_REFS;

  } hbHists, heHists, hfHists, hoHists;

};

#endif
