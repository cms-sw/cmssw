#ifndef DQM_HCALMONITORTASKS_HCALNZSMONITOR_H
#define DQM_HCALMONITORTASKS_HCALNZSMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"  // need for emap
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/TriggerResults.h"

class HcalNZSMonitor: public HcalBaseDQMonitor 
{
 public:
  HcalNZSMonitor(const edm::ParameterSet& ps);
  ~HcalNZSMonitor();


  void setup(DQMStore::IBooker &);
  void bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c);
  void analyze(edm::Event const&e, edm::EventSetup const&s);
  void processEvent(const FEDRawDataCollection& rawraw, const edm::TriggerResults&, int bxNum,const edm::TriggerNames & triggerNames);



  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void reset();

 private: 
  // Data accessors
  std::vector<int> selFEDs_;
   
  std::vector<std::string>  triggers_;
  int period_;
  
  //Monitoring elements
  MonitorElement* meFEDsizeVsLumi_;
  
  MonitorElement* meFEDsizesNZS_;
  MonitorElement* meUTCAFEDsizesNZS_;
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

  edm::InputTag rawdataLabel_;
  edm::InputTag hltresultsLabel_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  edm::EDGetTokenT<edm::TriggerResults> tok_res_;
};


#endif
