#ifndef HcalDetDiagLEDClient_GUARD_H
#define HcalDetDiagLEDClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
class TH2F;
class HcalDetDiagLEDClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalDetDiagLEDClient(){name_="";};
  HcalDetDiagLEDClient(std::string myname);//{ name_=myname;};
  HcalDetDiagLEDClient(std::string myname, const edm::ParameterSet& ps);

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
  
  void htmlOutput(DQMStore::IBooker &, DQMStore::IGetter &, std::string);
  bool validHtmlOutput(DQMStore::IBooker &, DQMStore::IGetter &);
 
  /// Destructor
  ~HcalDetDiagLEDClient();

 private:
  int nevts_;
  int status;
  TH2F *ChannelsLEDEnergy[4];
  TH2F *ChannelsLEDEnergyRef[4];
  TH2F *ChannelStatusMissingChannels[4];
  TH2F *ChannelStatusUnstableChannels[4];
  TH2F *ChannelStatusUnstableLEDsignal[4];
  TH2F *ChannelStatusLEDMean[4];
  TH2F *ChannelStatusLEDRMS[4];
  TH2F *ChannelStatusTimeMean[4];
  TH2F *ChannelStatusTimeRMS[4];
  double get_channel_status(std::string subdet,int eta,int phi,int depth,int type);
  double get_energy(std::string subdet,int eta,int phi,int depth,int type);

  // -- setup for problem cells
  bool doProblemCellSetup_; // default to true in the constructor

  // setup the problem cells 
  // this sets the doProblemCellSetup_ to false
  void setupProblemCells(DQMStore::IBooker &, DQMStore::IGetter &);
};

#endif
