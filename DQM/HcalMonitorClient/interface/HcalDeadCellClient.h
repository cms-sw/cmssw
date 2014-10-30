#ifndef HcalDeadCellClient_GUARD_H
#define HcalDeadCellClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HcalDeadCellClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalDeadCellClient(){name_="";};
  HcalDeadCellClient(std::string myname);//{ name_=myname;};
  HcalDeadCellClient(std::string myname, const edm::ParameterSet& ps);

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
  ~HcalDeadCellClient();

 private:
  int nevts_;

  int HBpresent_, HEpresent_, HOpresent_, HFpresent_;
  bool excludeHOring2_backup_; // this value is used for excludeHOring2 if it can't be read directly from the DQM file

  // -- setup the problem cells monitors
  bool doProblemCellSetup_; // defaults to true in the constructor

  // setup the problem cell monitors
  // Sets the doProblemCellSetup_ to false
  void setupProblemCells(DQMStore::IBooker &, DQMStore::IGetter &);

};

#endif
