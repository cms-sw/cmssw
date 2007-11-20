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
HcalDataFormatClient(const ParameterSet& ps, DaqMonitorBEInterface* dbe_);
HcalDataFormatClient();

/// Destructor
virtual ~HcalDataFormatClient();

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

  void resetAllME();
  void createTests();

private:

  void labelxBits(TH1F* hist);
  void labelyBits(TH2F* hist);
  
  int ievt_;
  int jevt_;
  
  bool cloneME_;
  bool debug_;
  string process_;
  string baseFolder_;

  //  MonitorUserInterface* mui_;
  DaqMonitorBEInterface* dbe_;
  
  bool subDetsOn_[4];

  TH1F* spigotErrs_;
  TH1F* badDigis_;
  TH1F* unmappedDigis_;
  TH1F* unmappedTPDs_;
  TH1F* fedErrMap_;
  TH1F* BCN_;

  TH1F* BCNCheck_;
  TH1F* EvtNCheck_;
  TH1F* FibOrbMsgBCN_;

  TH1F* dferr_[3];

  TH2F* BCNMap_;
  TH2F* EvtMap_;
  TH2F* ErrMapbyCrate_;
  TH2F* FWVerbyCrate_;
  TH2F* ErrCrate0_;
  TH2F* ErrCrate1_;
  TH2F* ErrCrate2_;
  TH2F* ErrCrate3_;
  TH2F* ErrCrate4_;
  TH2F* ErrCrate5_;
  TH2F* ErrCrate6_;
  TH2F* ErrCrate7_;
  TH2F* ErrCrate8_;
  TH2F* ErrCrate9_;
  TH2F* ErrCrate10_;
  TH2F* ErrCrate11_;
  TH2F* ErrCrate12_;
  TH2F* ErrCrate13_;
  TH2F* ErrCrate14_;
  TH2F* ErrCrate15_;
  TH2F* ErrCrate16_;
  TH2F* ErrCrate17_;

  /*
  TH2F* crateErrMap_[4];
  TH2F* fiberErrMap_[4];
  TH2F* spigotErrMap_[4];
  */

  // Quality criteria for data integrity
  float thresh_;
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

};

#endif
