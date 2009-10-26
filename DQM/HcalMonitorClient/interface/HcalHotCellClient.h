#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


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
  void beginJob();
  
  /// EndJob
  void endJob(std::map<HcalDetId, unsigned int>& myqual);
  
  /// BeginRun
  void beginRun(const EventSetup& c);
  
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

  void calculateProblems();

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
  
  bool dump2database_;

  // Histograms

  TH2F* AboveNeighborsHotCellsByDepth[4];
  TH2F* AboveEnergyThresholdCellsByDepth[4];
  TH2F* AbovePersistentThresholdCellsByDepth[4];
  TH2F* AbovePedestalHotCellsByDepth[4];

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

  TH2F* d_avgrechitenergymap[4];
  TH2F* d_avgrechitoccupancymap[4];
};

#endif
