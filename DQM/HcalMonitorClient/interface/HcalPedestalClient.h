
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

  void resetME();
  void createTests();


private:
  
  int ievt_;
  int jevt_;
  
  edm::ESHandle<HcalDbService> conditions_;

  bool collateSources_;
  bool cloneME_;
  bool verbose_;
  bool offline_;
  bool doPerChanTests_;
  string process_;
  
  MonitorUserInterface* mui_;
  HcalElectronicsMap* readoutMap_;

  TH1F* all_peds[3];
  TH1F* ped_rms[3];
  TH1F* ped_mean[3];
  TH1F* capid_mean[3];
  TH1F* capid_rms[3];
  TH1F* qie_mean[3];
  TH1F* qie_rms[3];
  //  TH1F* pedMapMean[2][4];
  //  TH1F* pedMapRMS[2][4];

  TH2F* err_map_geo[3];
  TH2F* err_map_elec[3];
  TH2F* geoRef;

  int etaMin[4]; int etaMax[4];
  int phiMin[4]; int phiMax[4];
  int depMin[4]; int depMax[4];

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
