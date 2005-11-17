#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

/** \class HcalRecHitMonitor
  *  
  * $Date: 2005/11/14 17:17:35 $
  * $Revision: 1.3 $
  * \author W. Fisher - FNAL
  */
class HcalRecHitMonitor: public HcalBaseMonitor {
public:
  HcalRecHitMonitor(); 
  ~HcalRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);


private:  ///Monitoring elements

  MonitorElement* m_meRECHIT_E_hb;
  MonitorElement* m_meRECHIT_T_hb;

  MonitorElement* m_meRECHIT_E_hf;
  MonitorElement* m_meRECHIT_T_hf;

  MonitorElement* m_meRECHIT_E_ho;
  MonitorElement* m_meRECHIT_T_ho;

};

#endif
