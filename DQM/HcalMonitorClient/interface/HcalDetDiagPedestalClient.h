#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGPEDESTALCLIENT_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGPEDESTALCLIENT_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

class HcalDetDiagPedestalClient : public HcalBaseClient {
public:
  HcalDetDiagPedestalClient();
  ~HcalDetDiagPedestalClient();
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
  
  TH1F *PedestalsAve4HB;
  TH1F *PedestalsAve4HE;
  TH1F *PedestalsAve4HO;
  TH1F *PedestalsAve4HF;
  TH1F *PedestalsAve4Simp;
  TH1F *PedestalsAve4ZDC;
  
  TH1F *PedestalsRefAve4HB;
  TH1F *PedestalsRefAve4HE;
  TH1F *PedestalsRefAve4HO;
  TH1F *PedestalsRefAve4HF;
  TH1F *PedestalsRefAve4Simp;
  TH1F *PedestalsRefAve4ZDC;
  
  TH1F *PedestalsAve4HBref;
  TH1F *PedestalsAve4HEref;
  TH1F *PedestalsAve4HOref;
  TH1F *PedestalsAve4HFref;
   
  TH1F *PedestalsRmsHB;
  TH1F *PedestalsRmsHE;
  TH1F *PedestalsRmsHO;
  TH1F *PedestalsRmsHF;
  TH1F *PedestalsRmsSimp;
  TH1F *PedestalsRmsZDC;
   
  TH1F *PedestalsRmsRefHB;
  TH1F *PedestalsRmsRefHE;
  TH1F *PedestalsRmsRefHO;
  TH1F *PedestalsRmsRefHF;
  TH1F *PedestalsRmsRefSimp;
  TH1F *PedestalsRmsRefZDC;
  
  TH1F *PedestalsRmsHBref;
  TH1F *PedestalsRmsHEref;
  TH1F *PedestalsRmsHOref;
  TH1F *PedestalsRmsHFref;  
 
  TH2F *Pedestals2DRmsHBHEHF;
  TH2F *Pedestals2DRmsHO;
  TH2F *Pedestals2DHBHEHF;
  TH2F *Pedestals2DHO;
  TH2F *Pedestals2DErrorHBHEHF;
  TH2F *Pedestals2DErrorHO;
   
  // Channel status
  double get_channel_status(char *subdet,int eta,int phi,int depth,int type);
  TH2F* ChannelStatusMissingChannels[6];
  TH2F* ChannelStatusUnstableChannels[6];
  TH2F* ChannelStatusBadPedestalMean[6];
  TH2F* ChannelStatusBadPedestalRMS[6];
};

#endif
