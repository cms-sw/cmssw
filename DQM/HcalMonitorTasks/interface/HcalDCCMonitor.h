#ifndef DQM_HCALMONITORTASKS_HCALDCCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDCCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"
#include "EventFilter/HcalRawToDigi/interface/HcalRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "CondFormats/HcalMapping/interface/HcalMappingTextFileReader.h"

/** \class HcalDCCMonitor
  *  
  * $Date: $
  * $Revision: $
  * \author W. Fisher - FNAL
  */
class HcalDCCMonitor: public HcalBaseMonitor {
public:
  HcalDCCMonitor(); 
  ~HcalDCCMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void done(int mode);
  void processEvent(edm::Handle<FEDRawDataCollection> rawraw);
  void unpack(const FEDRawData& raw, int a, int b, int c);


private: /// Data accessors
  std::auto_ptr<HcalMapping> m_readoutMap;
  std::string m_readoutMapSource;
  std::vector<int> m_fedUnpackList;
  int m_firstFED;

private:  ///Monitoring elements
  MonitorElement* m_meDCC_ERRWD;
  MonitorElement* m_meDCC_FMT;
  
};

#endif
