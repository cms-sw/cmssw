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
  void getHistograms();
  void loadHistograms(TFile* f);
  
  ///process report
  void report();
  
  void resetAllME();
  void createTests();

private:
  
  void generateBadChanList(string dir);
  vector<int> badChan_;
  vector<double> badMean_;
  vector<double> badRMS_;
  
  edm::ESHandle<HcalDbService> conditions_;
  const HcalElectronicsMap* readoutMap_;
  
  bool doPerChanTests_;
  bool plotPedRAW_;

  int nCrates_;
  TH1F* htrMean_[1000];
  TH1F* htrRMS_[1000];

  TH1F* all_peds_[4];
  TH1F* ped_rms_[4];
  TH1F* ped_mean_[4];

  TH1F* sub_rms_[4];
  TH1F* sub_mean_[4];

  TH1F* capid_mean_[4];
  TH1F* capid_rms_[4];
  TH1F* qie_mean_[4];
  TH1F* qie_rms_[4];

  TH2F* pedMapMeanD_[4];
  TH2F* pedMapRMSD_[4];

  TH2F* pedMapMean_E[4];
  TH2F* pedMapRMS_E[4];

  TH2F* err_map_geo_[4];
  TH2F* err_map_elec_[4];
  TH2F* geoRef_;

  // Quality criteria for data integrity
  float pedrms_thresh_;
  float pedmean_thresh_;
  float caprms_thresh_;
  float capmean_thresh_;
  
};

#endif
