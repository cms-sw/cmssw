#ifndef GUARD_HcalRecHitClient_H
#define GUARD_HcalRecHitClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalRecHitClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalRecHitClient();
  /// Destructor
  ~HcalRecHitClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  //void beginJob(const EventSetup& c);
  void beginJob();

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
  bool rechitclient_makeDiagnostics_;

  int rechitclient_checkNevents_;
  
  // Histograms
  TH2F* ProblemRecHits;
  TH2F* ProblemRecHitsByDepth[4];
  TH2F* OccupancyByDepth[4];
  TH2F* OccupancyThreshByDepth[4];
  TH2F* EnergyByDepth[4];
  TH2F* EnergyThreshByDepth[4];
  TH2F* TimeByDepth[4];
  TH2F* TimeThreshByDepth[4];
  TH2F* SumEnergyByDepth[4];
  TH2F* SumEnergyThreshByDepth[4];
  TH2F* SumTimeByDepth[4];
  TH2F* SumTimeThreshByDepth[4];

  // diagnostic histograms
  TH1F* d_HBEnergy;
  TH1F* d_HBTotalEnergy;
  TH1F* d_HBTime;
  TH1F* d_HBOccupancy;
  TH1F* d_HBThreshEnergy;
  TH1F* d_HBThreshTotalEnergy;
  TH1F* d_HBThreshTime;
  TH1F* d_HBThreshOccupancy;

  TH1F* d_HEEnergy;
  TH1F* d_HETotalEnergy;
  TH1F* d_HETime;
  TH1F* d_HEOccupancy;
  TH1F* d_HEThreshEnergy;
  TH1F* d_HEThreshTotalEnergy;
  TH1F* d_HEThreshTime;
  TH1F* d_HEThreshOccupancy;

  TH1F* d_HOEnergy;
  TH1F* d_HOTotalEnergy;
  TH1F* d_HOTime;
  TH1F* d_HOOccupancy;
  TH1F* d_HOThreshEnergy;
  TH1F* d_HOThreshTotalEnergy;
  TH1F* d_HOThreshTime;
  TH1F* d_HOThreshOccupancy;

  TH1F* d_HFEnergy;
  TH1F* d_HFTotalEnergy;
  TH1F* d_HFTime;
  TH1F* d_HFOccupancy;
  TH1F* d_HFThreshEnergy;
  TH1F* d_HFThreshTotalEnergy;
  TH1F* d_HFThreshTime;
  TH1F* d_HFThreshOccupancy;

  TH1F* h_HBEnergy_1D;
  TH1F* h_HEEnergy_1D;
  TH1F* h_HOEnergy_1D;
  TH1F* h_HFEnergy_1D;

  TH1F* h_HBEnergyRMS_1D;
  TH1F* h_HEEnergyRMS_1D;
  TH1F* h_HOEnergyRMS_1D;
  TH1F* h_HFEnergyRMS_1D;
};


#endif
