#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalDeadCellClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalDeadCellClient();
  /// Destructor
  ~HcalDeadCellClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob(const EventSetup& c);
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun(void);
  
  /// EndRun
  void endRun(void);
  
  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
  /// HtmlOutput
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlExpertOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  ///process report
  void report();
  
  void resetAllME();
  void createTests();

private:
  
  vector <std::string> subdets_;

  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool deadclient_makeDiagnostics_;

  bool deadclient_test_occupancy_;
  bool deadclient_test_rechit_occupancy_;
  bool deadclient_test_pedestal_;
  bool deadclient_test_neighbor_;
  bool deadclient_test_energy_;

  int deadclient_checkNevents_;
  int deadclient_checkNevents_occupancy_;
  int deadclient_checkNevents_rechit_occupancy_;
  int deadclient_checkNevents_pedestal_;
  int deadclient_checkNevents_neighbor_;
  int deadclient_checkNevents_energy_;

  // Histograms
  TH2F* ProblemDeadCells;
  TH2F* ProblemDeadCellsByDepth[6];
  TH2F* UnoccupiedDeadCellsByDepth[6];
  TH2F* UnoccupiedRecHitsByDepth[6];
  TH2F* BelowPedestalDeadCellsByDepth[6];
  TH2F* BelowNeighborsDeadCellsByDepth[6];
  TH2F* BelowEnergyThresholdCellsByDepth[6];

  // diagnostic histograms
  TH1F* d_HBnormped;
  TH1F* d_HEnormped;
  TH1F* d_HOnormped;
  TH1F* d_HFnormped;
  
  TH1F* d_HBrechitenergy;
  TH1F* d_HErechitenergy;
  TH1F* d_HOrechitenergy;
  TH1F* d_HFrechitenergy;
  
  TH2F* d_HBenergyVsNeighbor;
  TH2F* d_HEenergyVsNeighbor;
  TH2F* d_HOenergyVsNeighbor;
  TH2F* d_HFenergyVsNeighbor;
};

#endif
