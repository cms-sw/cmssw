#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGLASERCLIENT_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGLASERCLIENT_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondTools/Hcal/interface/HcalLogicalMapGenerator.h"
#include "CondTools/Hcal/interface/HcalLogicalMap.h"

class HcalDetDiagLaserClient : public HcalBaseClient {
public:
  HcalDetDiagLaserClient();
  ~HcalDetDiagLaserClient();
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
  void report();
  /// HtmlOutput
  bool haveOutput();
  int  SummaryStatus();
  void htmlOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests(); 
private:
  int status;
  TH1F *Energy;
  TH1F *Timing;
  TH1F *EnergyRMS;
  TH1F *TimingRMS;
  TH2F *Time2Dhbhehf;
  TH2F *Time2Dho;
  TH2F *Energy2Dhbhehf;
  TH2F *Energy2Dho;
};

#endif
