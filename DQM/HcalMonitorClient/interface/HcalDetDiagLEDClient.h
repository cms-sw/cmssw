#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGLEDCLIENT_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGLEDCLIENT_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

class HcalDetDiagLEDClient : public HcalBaseClient {
public:
  HcalDetDiagLEDClient();
  ~HcalDetDiagLEDClient();
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
  
  TH1F *Energy;
  TH1F *Timing;
  TH1F *EnergyRMS;
  TH1F *TimingRMS;
  TH1F *EnergyHF;
  TH1F *TimingHF;
  TH1F *EnergyRMSHF;
  TH1F *TimingRMSHF;
  TH1F *EnergyCorr;
  TH2F *Time2Dhbhehf;
  TH2F *Time2Dho;
  TH2F *Energy2Dhbhehf;
  TH2F *Energy2Dho;
  
  TH2F *HBPphi;
  TH2F *HBMphi;
  TH2F *HEPphi;
  TH2F *HEMphi;
  TH2F *HFPphi;
  TH2F *HFMphi;
  TH2F *HO0phi;
  TH2F *HO1Pphi;
  TH2F *HO2Pphi;
  TH2F *HO1Mphi;
  TH2F *HO2Mphi;
  
  TH2F* ChannelsLEDEnergy[6];
  TH2F* ChannelsLEDEnergyRef[6];
  double get_energy(char *subdet,int eta,int phi,int depth,int type);
  // Channel status
  double get_channel_status(char *subdet,int eta,int phi,int depth,int type);
  TH2F* ChannelStatusMissingChannels[6];
  TH2F* ChannelStatusUnstableChannels[6];
  TH2F* ChannelStatusUnstableLEDsignal[6];
  TH2F* ChannelStatusLEDMean[6];
  TH2F* ChannelStatusLEDRMS[6];
  TH2F* ChannelStatusTimeMean[6];
  TH2F* ChannelStatusTimeRMS[6];
};

#endif
