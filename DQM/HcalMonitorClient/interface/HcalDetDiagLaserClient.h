#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGLASERCLIENT_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGLASERCLIENT_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

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
  string ref_run;
  
  TH1F *hbheEnergy;
  TH1F *hbheTiming;
  TH1F *hbheEnergyRMS;
  TH1F *hbheTimingRMS;
  TH1F *hoEnergy;
  TH1F *hoTiming;
  TH1F *hoEnergyRMS;
  TH1F *hoTimingRMS;
  TH1F *hfEnergy;
  TH1F *hfTiming;
  TH1F *hfEnergyRMS;
  TH1F *hfTimingRMS;
  
  TH2F *Time2Dhbhehf;
  TH2F *Time2Dho;
  TH2F *Energy2Dhbhehf;
  TH2F *Energy2Dho;
  TH2F *refTime2Dhbhehf;
  TH2F *refTime2Dho;
  TH2F *refEnergy2Dhbhehf;
  TH2F *refEnergy2Dho;
  
  TH1F *Raddam[56];
};

#endif
