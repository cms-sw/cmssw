#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

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

struct HotCellHists{
  int type;
  int thresholds;
  TH2F* OCC_MAP_GEO_Max;
  TH2F* EN_MAP_GEO_Max;
  TH1F* MAX_E;
  TH1F* MAX_T;
  TH1F* MAX_ID;
  std::vector<TH2F*> OCCmap;
  std::vector<TH2F*> ENERGYmap;
  // NADA histograms
  TH2F* NADA_OCC_MAP;
  TH2F* NADA_EN_MAP;
  TH1F* NADA_NumHotCells;
  TH1F* NADA_testcell;
  TH1F* NADA_Energy;
  TH1F* NADA_NumNegCells;
  TH2F* NADA_NEG_OCC_MAP;
  TH2F* NADA_NEG_EN_MAP;
};

class HcalHotCellClient{

public:

/// Constructor
HcalHotCellClient(const ParameterSet& ps, DaqMonitorBEInterface* dbe_);
HcalHotCellClient();

/// Destructor
virtual ~HcalHotCellClient();

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


 ///process report
  void report();
  
  /// WriteDB
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlSubDetOutput(HotCellHists& hist, int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  void errorOutput();
  void getErrors(map<string, vector<QReport*> > out1, map<string, vector<QReport*> > out2, map<string, vector<QReport*> > out3);
  bool hasErrors() const { return dqmReportMapErr_.size(); }
  bool hasWarnings() const { return dqmReportMapWarn_.size(); }
  bool hasOther() const { return dqmReportMapOther_.size(); }

  void resetAllME();

  void createTests();
  void createSubDetTests(HotCellHists& hist);

  // Clear histograms
  void clearHists(HotCellHists& hist);
  void deleteHists(HotCellHists& hist);

  void getSubDetHistograms(HotCellHists& hist);
  void resetSubDetHistograms(HotCellHists& hist);
  void getSubDetHistogramsFromFile(HotCellHists& hist, TFile* infile);

private:

  int ievt_;
  int jevt_;

  bool collateSources_;
  bool cloneME_;
  bool verbose_;
  string process_;
  string baseFolder_;

  // Can we get threshold information from same .cfi file that HotCellMonitor uses?
  
  int thresholds_;



  //  MonitorUserInterface* mui_;
  DaqMonitorBEInterface* dbe_;

  bool subDetsOn_[4];

  TH2F* gl_geo_[4];
  TH2F* gl_en_[4];

  TH2F* occ_geo_[4][2];
  TH2F* occ_en_[4][2];
  TH1F* max_en_[4];
  TH1F* max_t_[4];
  
  // Quality criteria for data integrity
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;


  HotCellHists hbhists;
  HotCellHists hehists;
  HotCellHists hohists;
  HotCellHists hfhists;


  ofstream htmlFile;
};

#endif
