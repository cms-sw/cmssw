#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalHotCellClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalHotCellClient();
  /// Destructor
  ~HcalHotCellClient();

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

  // Introduce temporary error/warning checks
  bool hasErrors_Temp();
  bool hasWarnings_Temp();
  bool hasOther_Temp() {return false;}

private:
  
  vector <std::string> subdets_;

  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool hotclient_makeDiagnostics_;

  bool hotclient_test_persistent_;
  bool hotclient_test_pedestal_;
  bool hotclient_test_neighbor_;
  bool hotclient_test_energy_;

  int hotclient_checkNevents_;
  int hotclient_checkNevents_pedestal_;
  int hotclient_checkNevents_neighbor_;
  int hotclient_checkNevents_energy_;
  int hotclient_checkNevents_persistent_;

  // Histograms
  TH2F* ProblemHotCells;
  TH2F* ProblemHotCellsByDepth[6];

  TH2F* AboveNeighborsHotCellsByDepth[6];
  TH2F* AboveEnergyThresholdCellsByDepth[6];
  TH2F* AbovePersistentThresholdCellsByDepth[6];
  TH2F* AbovePedestalHotCellsByDepth[6];

  // diagnostic histograms
  TH1F* d_HBnormped;
  TH1F* d_HEnormped;
  TH1F* d_HOnormped;
  TH1F* d_HFnormped;
  TH1F* d_ZDCnormped;
  
  TH1F* d_HBrechitenergy;
  TH1F* d_HErechitenergy;
  TH1F* d_HOrechitenergy;
  TH1F* d_HFrechitenergy;
  TH1F* d_ZDCrechitenergy;

  TH2F* d_HBenergyVsNeighbor;
  TH2F* d_HEenergyVsNeighbor;
  TH2F* d_HOenergyVsNeighbor;
  TH2F* d_HFenergyVsNeighbor;
  TH2F* d_ZDCenergyVsNeighbor;

  TH2F* d_avgrechitenergymap[6];
};

#endif
