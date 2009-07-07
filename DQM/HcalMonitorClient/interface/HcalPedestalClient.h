#ifndef HcalPedestalClient_H
#define HcalPedestalClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include <CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalPedestalClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalPedestalClient();
  /// Destructor
  ~HcalPedestalClient();

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
  bool hasOther_Temp()  {return false;}


private:
  
  //void generateBadChanList(string dir);
  vector<int> badChan_;
  vector<double> badMean_;
  vector<double> badRMS_;
  
  vector <std::string> subdets_;

  bool doFCpeds_; // pedestal units in fC (if false, assume ADC)
  // specify time slices over which to calculate pedestals -- are these needed in client?
  bool startingTimeSlice_;
  bool endingTimeSlice_;

  // Specify maximum allowed difference between ADC pedestal and nominal value
  double nominalPedMeanInADC_;
  double nominalPedWidthInADC_;
  double maxPedMeanDiffADC_;
  double maxPedWidthDiffADC_; // specify maximum width of pedestal (in ADC)
  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool makeDiagnostics_;
  TH2F* MeanMapByDepth[6];
  TH2F* RMSMapByDepth[6];

 // Problem Pedestal Plots
  TH2F* ProblemPedestals;
  TH2F* ProblemPedestalsByDepth[6];

  // Pedestals from Database
  TH2F* ADC_PedestalFromDBByDepth[6];
  TH2F* ADC_WidthFromDBByDepth[6];
  TH2F* fC_PedestalFromDBByDepth[6];
  TH2F* fC_WidthFromDBByDepth[6];
  TH1F* ADC_PedestalFromDBByDepth_1D[6];
  TH1F* ADC_WidthFromDBByDepth_1D[6];
  TH1F* fC_PedestalFromDBByDepth_1D[6];
  TH1F* fC_WidthFromDBByDepth_1D[6];

  // Raw pedestals in ADC
  TH2F* rawADCPedestalMean[6];
  TH2F* rawADCPedestalRMS[6];
  TH1F* rawADCPedestalMean_1D[6];
  TH1F* rawADCPedestalRMS_1D[6];
  
  // subtracted pedestals in ADC
  TH2F* subADCPedestalMean[6];
  TH2F* subADCPedestalRMS[6];
  TH1F* subADCPedestalMean_1D[6];
  TH1F* subADCPedestalRMS_1D[6];
  
  // Raw pedestals in FC
  TH2F* rawfCPedestalMean[6];
  TH2F* rawfCPedestalRMS[6];
  TH1F* rawfCPedestalMean_1D[6];
  TH1F* rawfCPedestalRMS_1D[6];
  
  // subtracted pedestals in FC
  TH2F* subfCPedestalMean[6];
  TH2F* subfCPedestalRMS[6];
  TH1F* subfCPedestalMean_1D[6];
  TH1F* subfCPedestalRMS_1D[6];

};

#endif
