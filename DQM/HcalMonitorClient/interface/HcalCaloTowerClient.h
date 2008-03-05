#ifndef HcalCaloTowerClient_H
#define HcalCaloTowerClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HcalCaloTowerClient : public HcalBaseClient {

 public:
  
  /// Constructor
  HcalCaloTowerClient();
  
  /// Destructor
  ~HcalCaloTowerClient();
  
  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);    

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob(void);
  
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
  void getHistograms();
  void loadHistograms(TFile* f);

  void report();

  void resetAllME();
  void createTests();
 private:
  // Basic occupancy, energy histograms
  TH2F* occ_;
  TH2F* energy_;
};

#endif
