#ifndef HcalDataFormatClient_H
#define HcalDataFormatClient_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

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

class HcalDataFormatClient{

public:

/// Constructor
HcalDataFormatClient(const ParameterSet& ps, MonitorUserInterface* mui);
HcalDataFormatClient();

/// Destructor
virtual ~HcalDataFormatClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

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
  
  void errorOutput();
  void getErrors(map<string, vector<QReport*> > out1, map<string, vector<QReport*> > out2, map<string, vector<QReport*> > out3);
  bool hasErrors() const { return dqmReportMapErr_.size(); }
  bool hasWarnings() const { return dqmReportMapWarn_.size(); }
  bool hasOther() const { return dqmReportMapOther_.size(); }

  void resetME();
  void createTests();

private:

  void labelBits(TH1F* hist);
  
  int ievt_;
  int jevt_;
  
  bool cloneME_;
  bool verbose_;
  string process_;

  MonitorUserInterface* mui_;

  bool subDetsOn_[4];

  TH1F* spigotErrs_;
  TH1F* badDigis_;
  TH1F* unmappedDigis_;
  TH1F* unmappedTPDs_;
  TH1F* fedErrMap_;

  TH1F* dferr_[4];
  TH2F* crateErrMap_[4];
  TH2F* fiberErrMap_[4];
  TH2F* spigotErrMap_[4];

  // Quality criteria for data integrity
  float thresh_;
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

};

#endif
