#ifndef HcalRecHitClient_H
#define HcalRecHitClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

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

  TH2F* tot_occ_[4];
  TH1F* tot_energy_;

  TH2F* occ_[4];
  TH1F* energy_[4];
  TH1F* energyT_[4];
  TH1F* time_[4];

  TH1F* hfshort_E_all;
  //TH1F* hfshort_E_low;
  TH1F* hfshort_T_all;

};

#endif
