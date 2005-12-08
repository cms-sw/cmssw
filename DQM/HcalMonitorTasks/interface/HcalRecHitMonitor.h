#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalRecHitMonitor
  *  
  * $Date: 2005/11/30 22:05:36 $
  * $Revision: 1.2 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);


private:  ///Monitoring elements

  bool m_doPerChannel;

  MonitorElement* m_meRECHIT_E_all;

  MonitorElement* m_meRECHIT_E_hb_all;
  MonitorElement* m_meRECHIT_E_hb_tot;
  MonitorElement* m_meRECHIT_T_hb_tot;

  MonitorElement* m_meRECHIT_E_hf_all;
  MonitorElement* m_meRECHIT_E_hf_tot;
  MonitorElement* m_meRECHIT_T_hf_tot;

  MonitorElement* m_meRECHIT_E_ho_all;
  MonitorElement* m_meRECHIT_E_ho_tot;
  MonitorElement* m_meRECHIT_T_ho_tot;

  std::map<HcalDetId, MonitorElement*> m_meRECHIT_E_hb, m_meRECHIT_T_hb;  // complicated per-channel histogram setup
  std::map<HcalDetId, MonitorElement*> m_meRECHIT_E_hf, m_meRECHIT_T_hf;  // complicated per-channel histogram setup
  std::map<HcalDetId, MonitorElement*> m_meRECHIT_E_ho, m_meRECHIT_T_ho;  // complicated per-channel histogram setup

};

#endif
