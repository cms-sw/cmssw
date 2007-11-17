
#ifndef HcalLEDClient_H
#define HcalLEDClient_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace edm;
using namespace std;

class HcalLEDClient{
  
public:
  
  /// Constructor
  HcalLEDClient(const ParameterSet& ps, MonitorUserInterface* mui);
  HcalLEDClient();
  
  /// Destructor
  virtual ~HcalLEDClient();
  
  /// Subscribe/Unsubscribe to Monitoring Elements
  void subscribe(void);
  void subscribeNew(void);
  void unsubscribe(void);
  
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
  
  void errorOutput();
  void getErrors(map<string, vector<QReport*> > out1, map<string, vector<QReport*> > out2, map<string, vector<QReport*> > out3);
  bool hasErrors() const { return dqmReportMapErr_.size(); }
  bool hasWarnings() const { return dqmReportMapWarn_.size(); }
  bool hasOther() const { return dqmReportMapOther_.size(); }

  void resetAllME();
  void createTests();


private:
  
  int ievt_;
  int jevt_;

  bool collateSources_;
  bool cloneME_;
  bool verbose_;
  string process_;
  string m_outputFileName;
  ofstream m_outTextFile;

  MonitorUserInterface* mui_;
  const HcalElectronicsMap* readoutMap_;
  edm::ESHandle<HcalDbService> conditions_;

  bool subDetsOn_[4];

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

  // Quality criteria for data integrity
  float rms_thresh_;
  float mean_thresh_;
  
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;
};

#endif
