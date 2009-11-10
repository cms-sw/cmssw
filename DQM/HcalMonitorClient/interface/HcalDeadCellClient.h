#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

class HcalDeadCellClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalDeadCellClient();
  /// Destructor
  ~HcalDeadCellClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  void calculateProblems(void); // calculates problem histogram contents

  /// BeginJob
  void beginJob();
  
  /// EndJob
  void endJob();


  /// BeginRun
  void beginRun(const EventSetup& c);

  /// EndRun
  void endRun(std::map<HcalDetId, unsigned int>& myqual);
  
  /// Update Channel Status -- called by EndRun
  void updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual);

  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
  /// HtmlOutput
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlExpertOutput(int run, string htmlDir, string htmlName);
  void getHistograms(bool getall=false);
  void loadHistograms(TFile* f);
  
  ///process report
  void report();
  
  void resetAllME();
  void createTests();

  // Introduce temporary error/warning checks
  bool hasErrors_Temp();
  bool hasWarnings_Temp();
  bool hasOther_Temp() {return false;}
private:
  
  vector <std::string> subdets_;

  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool deadclient_makeDiagnostics_;

  bool deadclient_test_digis_;
  bool deadclient_test_rechits_;
  bool dump2database_;

  int deadclient_checkNevents_;

  // Histograms

  TH2F* RecentMissingDigisByDepth[4];
  TH2F* DigiPresentByDepth[4];
  TH2F* RecHitsPresentByDepth[4];
  TH2F* RecentMissingRecHitsByDepth[4];

  TProfile* ProblemsVsLB;
  TProfile* ProblemsVsLB_HB;
  TProfile* ProblemsVsLB_HE;
  TProfile* ProblemsVsLB_HO;
  TProfile* ProblemsVsLB_HF;

  TProfile* NumberOfNeverPresentDigis;
  TProfile* NumberOfNeverPresentDigisHB;
  TProfile* NumberOfNeverPresentDigisHE;
  TProfile* NumberOfNeverPresentDigisHO;
  TProfile* NumberOfNeverPresentDigisHF;

  TProfile* NumberOfRecentMissingDigis;
  TProfile* NumberOfRecentMissingDigisHB;
  TProfile* NumberOfRecentMissingDigisHE;
  TProfile* NumberOfRecentMissingDigisHO;
  TProfile* NumberOfRecentMissingDigisHF;


  TProfile* NumberOfNeverPresentRecHits;
  TProfile* NumberOfNeverPresentRecHitsHB;
  TProfile* NumberOfNeverPresentRecHitsHE;
  TProfile* NumberOfNeverPresentRecHitsHO;
  TProfile* NumberOfNeverPresentRecHitsHF;

  TProfile* NumberOfRecentMissingRecHits;
  TProfile* NumberOfRecentMissingRecHitsHB;
  TProfile* NumberOfRecentMissingRecHitsHE;
  TProfile* NumberOfRecentMissingRecHitsHO;
  TProfile* NumberOfRecentMissingRecHitsHF;

};

#endif
