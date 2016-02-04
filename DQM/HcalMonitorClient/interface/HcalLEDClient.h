#ifndef HcalLEDClient_H
#define HcalLEDClient_H

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

class HcalLEDClient : public HcalBaseClient {
  
public:
  
  /// Constructor
  HcalLEDClient();
  /// Destructor
  ~HcalLEDClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob();
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun(const EventSetup& c);
  
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
  
  string m_outputFileName;
  ofstream m_outTextFile;

  const HcalElectronicsMap* readoutMap_;
  edm::ESHandle<HcalDbService> conditions_;

  TH1F* avg_shape_[4];
  TH1F* avg_time_[4];
  TH1F* avg_energy_[4];

  TH1F* rms_shape_[4];
  TH1F* mean_shape_[4];
  TH1F* rms_time_[4];
  TH1F* mean_time_[4];
  TH1F* rms_energy_[4];
  TH1F* mean_energy_[4];

  TH2F* err_map_geo_[4];
  TH2F* err_map_elec_[4];

  TH2F* rms_energyDep_[4];
  TH2F* mean_energyDep_[4];
  TH2F* rms_timeDep_[4];
  TH2F* mean_timeDep_[4];
  TH2F* rms_shapeDep_[4];
  TH2F* mean_shapeDep_[4];

  map<unsigned int, TH2F*> rms_energyElec_;
  map<unsigned int, TH2F*> mean_energyElec_;
  map<unsigned int, TH2F*> rms_timeElec_;
  map<unsigned int, TH2F*> mean_timeElec_;
  map<unsigned int, TH2F*> rms_shapeElec_;
  map<unsigned int, TH2F*> mean_shapeElec_;


  TH1F* HFlumi_etsum;
  TH1F* HFlumi_occabthr1;
  TH1F* HFlumi_occbetthr1;
  TH1F* HFlumi_occbelthr1;
  TH1F* HFlumi_occabthr2;
  TH1F* HFlumi_occbetthr2;
  TH1F* HFlumi_occbelthr2;



  // Quality criteria for data integrity
  float rms_thresh_;
  float mean_thresh_;

};

#endif
