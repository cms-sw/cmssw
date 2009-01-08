#ifndef HcalDigiClient_H
#define HcalDigiClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

struct DigiClientHists
{
  // structures of histograms for each subdetector

  TH1F* shape;
  TH1F* shapeThresh;
  TH1F* presample;
  TH1F* BQ;
  TH1F* BQFrac;
  TH1F* DigiFirstCapID;
  TH1F* DVerr;
  TH1F* CapID;
  TH1F* ADC;
  TH1F* ADCsum;
  TH1F* TS_sum_plus[9];
  TH1F* TS_sum_minus[9];
};

class HcalDigiClient : public HcalBaseClient {

 public:
  
  /// Constructor
  HcalDigiClient();
  
  /// Destructor
  ~HcalDigiClient();

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
  
  
  ///process report
  void report();
  
  /// WriteDB
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlExpertOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests();

private:

  vector <std::string> subdets_;
  bool digiclient_makeDiagnostics_;
  int digiclient_checkNevents_;
  

  // Histograms
  TH2F* ProblemDigis;
  TH2F* ProblemDigisByDepth[6];
  TH2F* DigiErrorsBadCapID[6];
  TH2F* DigiErrorsBadDigiSize[6];
  TH2F* DigiErrorsBadADCSum[6];
  TH2F* DigiErrorsNoDigi[6];
  TH2F* DigiErrorsDVErr[6];
  TH2F* DigiOccupancyByDepth[6];
  TH2F* DigiOccupancyVME;
  TH2F* DigiOccupancySpigot;
  TH2F* DigiErrorEtaPhi;
  TH2F* DigiErrorVME;
  TH2F* DigiErrorSpigot;

  TH2F* DigiSize;
  TH1F* DigiOccupancyEta;
  TH1F* DigiOccupancyPhi;
  TH1F* DigiNum;
  TH1F* DigiBQ;
  TH1F* DigiBQFrac;

  DigiClientHists hbHists, heHists, hoHists, hfHists;
};

#endif
