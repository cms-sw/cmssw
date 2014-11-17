#ifndef HcalDetDiagTimingClient_GUARD_H
#define HcalDetDiagTimingClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HcalDetDiagTimingClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalDetDiagTimingClient(){name_="";};
  HcalDetDiagTimingClient(std::string myname);//{ name_=myname;};
  HcalDetDiagTimingClient(std::string myname, const edm::ParameterSet& ps);

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
  ~HcalDetDiagTimingClient();

 private:
  int nevts_;

  // - setup problem cell flags
  bool doProblemCellSetup_;  // defaults to true in the constructor
  // setup the problem cell monitor elements
  // This method sets the doProblemCellSetup_ flag to false
  void setupProblemCells(DQMStore::IBooker &, DQMStore::IGetter &);

};

#endif
