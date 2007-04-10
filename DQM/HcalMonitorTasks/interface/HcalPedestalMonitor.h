#ifndef DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H
#define DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

/** \class HcalPedestalMonitor
  *  
  * $Date: 2007/04/02 13:19:38 $
  * $Revision: 1.7 $
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

private: 
  void perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, MonitorElement*> > &toolP, map<HcalDetId, map<int, MonitorElement*> > &toolS);

  bool m_doPerChannel;
  map<HcalDetId, map<int,MonitorElement*> >::iterator _meo;

  string m_outputFile;
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;
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
  } hbHists, hfHists, hoHists;

};

#endif
