#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/** \class HcalDigiMonitor
  *  
  * $Date: 2005/11/14 16:48:07 $
  * $Revision: 1.2 $
  * \author W. Fisher - FNAL
  */
class HcalDigiMonitor: public HcalBaseMonitor {
public:
  HcalDigiMonitor(); 
  ~HcalDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf);

private:  ///Monitoring elements

  void fillErrors(const HBHEDataFrame& hb);
  void fillErrors(const HODataFrame& ho);
  void fillErrors(const HFDataFrame& hf);
  
  MonitorElement* m_meDIGI_SIZE_hb;
  MonitorElement* m_meDIGI_PRESAMPLE_hb;
  MonitorElement* m_meQIE_CAPID_hb;
  MonitorElement* m_meQIE_ADC_hb;
  MonitorElement* m_meQIE_DV_hb;
  MonitorElement* m_meERR_MAP_hb;

  MonitorElement* m_meDIGI_SIZE_hf;
  MonitorElement* m_meDIGI_PRESAMPLE_hf;
  MonitorElement* m_meQIE_CAPID_hf;
  MonitorElement* m_meQIE_ADC_hf;
  MonitorElement* m_meQIE_DV_hf;
  MonitorElement* m_meERR_MAP_hf;

  MonitorElement* m_meDIGI_SIZE_ho;
  MonitorElement* m_meDIGI_PRESAMPLE_ho;
  MonitorElement* m_meQIE_CAPID_ho;
  MonitorElement* m_meQIE_ADC_ho;
  MonitorElement* m_meQIE_DV_ho;
  MonitorElement* m_meERR_MAP_ho;
};

#endif
