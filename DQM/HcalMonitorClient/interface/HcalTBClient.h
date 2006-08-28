#ifndef HcalTBClient_H
#define HcalTBClient_H

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

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace edm;
using namespace std;

class HcalTBClient{

public:

/// Constructor
HcalTBClient(const ParameterSet& ps, MonitorUserInterface* mui);

/// Destructor
virtual ~HcalTBClient();

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

  void qadcHTML(string htmlDir, string htmlName);
  void timingHTML(string htmlDir, string htmlName);
  void evtposHTML(string htmlDir, string htmlName);
 
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
  
  bool cloneME_;
  bool verbose_;
  string process_;

  MonitorUserInterface* mui_;
  
  TH1F* CHK[3];
  TH1F* TOFQ[2];
  TH1F* TOFT_S[3];
  TH1F* TOFT_J[3];
  TH1F* DT[4];
  TH2F* WC[8];
  TH1F* WCX[8];
  TH1F* WCY[8];
  
  // Quality criteria for data integrity
  float thresh_;
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

};

#endif
