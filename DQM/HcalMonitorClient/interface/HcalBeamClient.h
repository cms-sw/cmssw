#ifndef HcalBeamClient_GUARD_H
#define HcalBeamClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HcalBeamClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalBeamClient(){name_="";};
  HcalBeamClient(std::string myname);//{ name_=myname;};
  HcalBeamClient(std::string myname, const edm::ParameterSet& ps);

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
  ~HcalBeamClient();

 private:
  int nevts_;

  // -- setup the problem cells monitor
  bool setupProblemCells_;   // defaults to true in constructor
  
  // perform the setup of the problem cells monitor elements and EtaPhiHists
  // This function sets the above setupProblemCells_ to false.
  void doProblemCellSetup(DQMStore::IBooker &, DQMStore::IGetter &);
  
};

#endif
