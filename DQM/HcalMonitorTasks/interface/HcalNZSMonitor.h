#ifndef DQM_HCALMONITORTASKS_HCALNZSMONITOR_H
#define DQM_HCALMONITORTASKS_HCALNZSMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/FWLite/interface/TriggerNames.h"
#include <math.h>

class HcalNZSMonitor: public HcalBaseMonitor {
 public:
  HcalNZSMonitor();
  ~HcalNZSMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEvent(const FEDRawDataCollection& rawraw, edm::TriggerResults, int bxNum);

  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();
  void reset();
  //void beginRun();
  void endLuminosityBlock();
  void UpdateMEs ();  

 private: 
  // Data accessors
  std::vector<int> selFEDs_;
   
  std::vector<std::string>  triggers_;
  int period_;
  
  //Monitoring elements
  MonitorElement* meFEDsizeVsLumi_;
  
  MonitorElement* meFEDsizesNZS_;
  MonitorElement* meL1evtNumber_;
  MonitorElement* meIsUS_;
  MonitorElement* meBXtriggered_;
  MonitorElement* meTrigFrac_;
  MonitorElement* meFullCMSdataSize_;

  bool isUnsuppressed (HcalHTRData& payload); //Return the US bit: ExtHdr7[bit 15]
  uint64_t UScount[32][15];

  int nAndAcc;
  int nAcc_Total;
  std::vector<int> nAcc;
};


#endif
