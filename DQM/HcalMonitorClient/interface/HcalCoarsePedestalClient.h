#ifndef HcalCoarsePedestalClient_GUARD_H
#define HcalCoarsePedestalClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HcalCoarsePedestalClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalCoarsePedestalClient(){name_="";};
  HcalCoarsePedestalClient(std::string myname);//{ name_=myname;};
  HcalCoarsePedestalClient(std::string myname, const edm::ParameterSet& ps);

  void analyze(DQMStore::IBooker &, DQMStore::IGetter &);
  void calculateProblems(DQMStore::IBooker &, DQMStore::IGetter &); // calculates problem histogram contents
  void updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual);
  void endJob(void);
  void beginRun(void);
  //void endRun(void); 
  void setup(void);  
  void cleanup(void);

  bool hasErrors_Temp(void);  
  bool hasWarnings_Temp(void);
  bool hasOther_Temp(void);
  bool test_enabled(void);
  
  /// Destructor
  ~HcalCoarsePedestalClient();

 private:
  int nevts_;

  double ADCDiffThresh_;
  TH2F* DatabasePedestalsADCByDepth[4];
  EtaPhiHists* CoarsePedestalsByDepth;
  MonitorElement* CoarsePedDiff;

  // -- setup problem cells monitors
  bool doCoarseSetup_;  // defaults to true in constructor

  // performs the setup of the coarse cell 
  // sets the doCoarseSetup_ to false  
  void setupCoarsePedestal(DQMStore::IBooker &, DQMStore::IGetter &);

};

#endif
