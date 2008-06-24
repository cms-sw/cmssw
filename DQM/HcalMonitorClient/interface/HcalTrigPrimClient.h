#ifndef HcalTrigPrimClient_H
#define HcalTrigPrimClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HcalTrigPrimClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalTrigPrimClient();
  
  /// Destructor
  ~HcalTrigPrimClient();

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
  void htmlOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests();

 private:
  
  TH1F* tpCount_;
  TH1F* tpCountThr_;
  TH1F* tpSize_;
  TH1F* tpSpectrum_[10];
  TH1F* tpSpectrumAll_;
  TH1F* tpETSumAll_;
  TH1F* tpSOI_ET_;
  TH1F* OCC_ETA_;
  TH1F* OCC_PHI_;
  TH2F* OCC_ELEC_VME_;
  TH2F* OCC_ELEC_DCC_;
  TH2F* OCC_MAP_GEO_;
  TH2F* OCC_MAP_SLB_;
  TH2F* OCC_MAP_THR_;
  TH1F* EN_ETA_;
  TH1F* EN_PHI_;
  TH2F* EN_ELEC_VME_;
  TH2F* EN_ELEC_DCC_;
  TH2F* EN_MAP_GEO_;
  TH1F* TPTiming_;
  TH1F* TPTimingTop_;
  TH1F* TPTimingBot_;
  TH1F* TP_ADC_;
  TH1F* MAX_ADC_;
  TH1F* TS_MAX_;
  TH2F* TPOcc_;
  TH2F* TPvsDigi_;


};

#endif
