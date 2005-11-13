#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/** \class HcalDigiMonitor
  *  
  * $Date: $
  * $Revision: $
  * \author W. Fisher - FNAL
  */
class HcalDigiMonitor: public HcalBaseMonitor {
public:
  HcalDigiMonitor(); 
  ~HcalDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void done(int mode);
  void processEvent(std::vector<edm::Handle<HBHEDigiCollection> > hbhe,
		    std::vector<edm::Handle<HODigiCollection> > ho,
		    std::vector<edm::Handle<HFDigiCollection> > hf);

  void fillErrors(const HBHEDataFrame);
  void fillErrors(const HODataFrame);
  void fillErrors(const HFDataFrame);

private:  ///Monitoring elements
  
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
