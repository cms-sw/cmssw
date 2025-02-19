#ifndef HcalLaserClient_H
#define HcalLaserClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include <CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HcalLaserClient : public HcalBaseClient {
  
 public:  
  HcalLaserClient();
  ~HcalLaserClient();

  void init( const edm::ParameterSet&, DQMStore*, const string );
  void setup(void);

  void beginJob();
  void beginRun(const EventSetup& c);
  void analyze(void);
  void endRun(void);
  void endJob(void);

  void cleanup(void);

  void htmlOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  void report();
  
  void resetAllME();
  void createTests();

 private:  
  TH1F* TDCNumChannels_;
  TH1F* TDCTrigger_;
  TH1F* TDCRawOptosync_;
  TH1F* TDCClockOptosync_;
  TH1F* TDCRawOptosync_Trigger_;

  TH1F* QADC_[32];

  TH1F* avg_shape_[4];
  TH1F* avg_time_[4];
  TH1F* avg_energy_[4];

  TH1F* rms_shape_[4];
  TH1F* mean_shape_[4];
  TH1F* rms_time_[4];
  TH1F* mean_time_[4];
  TH1F* rms_energy_[4];
  TH1F* mean_energy_[4];

  TH2F* rms_energyDep_[4];
  TH2F* mean_energyDep_[4];
  TH2F* rms_timeDep_[4];
  TH2F* mean_timeDep_[4];
  TH2F* rms_shapeDep_[4];
  TH2F* mean_shapeDep_[4];
};

#endif
