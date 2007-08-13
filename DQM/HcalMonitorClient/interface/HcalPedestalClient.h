
#ifndef HcalPedestalClient_H
#define HcalPedestalClient_H

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
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
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

class HcalPedestalClient{
  
public:
  
  /// Constructor
  HcalPedestalClient(const ParameterSet& ps, MonitorUserInterface* mui);
  HcalPedestalClient();
  
  /// Destructor
  virtual ~HcalPedestalClient();
  
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

  void generateBadChanList(string dir);
  vector<int> badChan_;
  vector<double> badMean_;
  vector<double> badRMS_;

  int ievt_;
  int jevt_;

  bool subDetsOn_[4];
  edm::ESHandle<HcalDbService> conditions_;

  bool collateSources_;
  bool cloneME_;
  bool verbose_;
  bool offline_;
  bool doPerChanTests_;
  bool plotPedRAW_;
  string process_;
  
  MonitorUserInterface* mui_;
  const HcalElectronicsMap* readoutMap_;

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
  
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;
};

#endif
