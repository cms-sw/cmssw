#ifndef DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H
#define DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

/** \class HcalPedestalMonitor
  *  
  * $Date: 2006/12/12 19:10:27 $
  * $Revision: 1.6 $
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
  void perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, MonitorElement*> > &tool);

  bool m_doPerChannel;
  map<HcalDetId, map<int,MonitorElement*> >::iterator _meo;

  string m_outputFile;
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;

  MonitorElement* meEVT_;
  int ievt_;
  
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  map<HcalDetId,bool> REG;

  MonitorElement* MEAN_MAP1;
  MonitorElement*  RMS_MAP1;

  MonitorElement* MEAN_MAP2;
  MonitorElement*  RMS_MAP2;

  MonitorElement* MEAN_MAP3;
  MonitorElement*  RMS_MAP3;

  MonitorElement* MEAN_MAP4;
  MonitorElement*  RMS_MAP4;

  struct{
    map<HcalDetId,map<int, MonitorElement*> > PEDVALS;
    MonitorElement* ALLPEDS;
    MonitorElement* PEDRMS;
    MonitorElement* PEDMEAN;    

    MonitorElement* CAPIDRMS;
    MonitorElement* CAPIDMEAN;    

    MonitorElement* QIERMS;
    MonitorElement* QIEMEAN;    

    MonitorElement* ERRGEO;
    MonitorElement* ERRELEC;    
  } hbHists, hfHists, hoHists;

};

#endif
