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
#include "TFile.h"
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
HcalTBClient();

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
  void loadHistograms(TFile* infile);
  void dumpHistograms(vector<TH1F*> &hist1, vector<TH2F*> &hist2);
  
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
  TH2F* WC[8];
  TH1F* WCX[8];
  TH1F* WCY[8];

  TH1F* TOF_DT[3];
  TH1F* TRIGT;
  TH1F* L1AT;
  TH1F* BCT;
  TH1F* PHASE;

  TH1F* HTIME[3];
  TH1F* HRES[3];
  TH2F* HPHASE[3];
  TH2F* ERES[3];
  
  /*
  meTOF_DT1_ = m_dbe->book1D("TOF TDC - Delta 1","TOF TDC - Delta 1",100,0,100);
  meTOF_DT2_ = m_dbe->book1D("TOF TDC - Delta 2","TOF TDC - Delta 2",100,0,100);
  meTOF_DT_ = m_dbe->book1D("TOF Time - Delta","TOF Time - Delta",100,0,100);
  
  meTrigger_ = m_dbe->book1D("Trigger Timing","Trigger Time",100,0,100);
  meTTCL1A_ = m_dbe->book1D("TTC L1A Timing","TTC L1A Time",100,0,100);
  meBeamCoinc_ = m_dbe->book1D("Beam Coincidence Timing","Beam Coincidence Time",100,0,100);
  
  meBeamPhase_ = m_dbe->book1D("TB Phase","TB Phase",100,0,100);
  meHBTime_ = m_dbe->book1D("HB Time","HB Time",200,-100,100);
  meHBEnergyRes_= m_dbe->book2D("HB Time Resolution vs Energy","HB Time Resolution vs Energy",300,0,300,200,0,100);
  meHBPhase_= m_dbe->book2D("HB Time vs Phase","HB Time vs Phase",100,0,100,200,-100,100);
  meHBTimeRes_= m_dbe->book1D("HB Time Resolution","HB Time Resolution",200,-100,200);
  
  meHOTime_ = m_dbe->book1D("HO Time","HO Time",200,-100,100);
  meHOEnergyRes_= m_dbe->book2D("HO Time Resolution vs Energy","HO Time Resolution vs Energy",300,0,300,200,-100,100);
  meHOPhase_= m_dbe->book2D("HO Time vs Phase","HO Time vs Phase",100,0,100,200,-100,100);
  meHOTimeRes_= m_dbe->book1D("HO Time Resolution","HO Time Resolution",200,-100,100);
  */

  // Quality criteria for data integrity
  float thresh_;
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

};

#endif
