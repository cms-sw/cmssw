#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

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

/*
using namespace cms;
using namespace edm;
using namespace std;
*/

struct DeadCellHists{
  int type;
  TH2F* deadADC_OccMap;
  TH1F* deadADC_Eta;
  TH2F* badCAPID_OccMap;
  TH1F* badCAPID_Eta;
  TH1F* ADCdist;
  TH2F* NADACoolCellMap;
  TH2F* digiCheck;
  TH2F* cellCheck;
  TH2F* AbovePed;
  TH2F* CoolCellBelowPed;
  std::vector<TH2F*> DeadCap;
};

class HcalDeadCellClient{

public:

/// Constructor
  HcalDeadCellClient(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe_);
HcalDeadCellClient();

/// Destructor
virtual ~HcalDeadCellClient();

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
  void htmlOutput(int run, std::string htmlDir, std::string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  void errorOutput();
  void getErrors(std::map<std::string, std::vector<QReport*> > out1, std::map<std::string, std::vector<QReport*> > out2, std::map<std::string, std::vector<QReport*> > out3);
  bool hasErrors() const { return dqmReportMapErr_.size(); }
  bool hasWarnings() const { return dqmReportMapWarn_.size(); }
  bool hasOther() const { return dqmReportMapOther_.size(); }

  void resetAllME();
  void createTests();
  void createSubDetTests(DeadCellHists& hist);

  // Clear histograms
  void clearHists(DeadCellHists& hist);
  void deleteHists(DeadCellHists& hist);

  void getSubDetHistograms(DeadCellHists& hist);
  void resetSubDetHistograms(DeadCellHists& hist);
  void getSubDetHistogramsFromFile(DeadCellHists& hist, TFile* infile);
  void htmlSubDetOutput(DeadCellHists& hist, int runNo, 
			std::string htmlDir, 
			std::string htmlName);

private:

  int ievt_;
  int jevt_;

  bool collateSources_;
  bool cloneME_;
  bool verbose_;
  std::string process_;
  std::string baseFolder_;

  DaqMonitorBEInterface* dbe_;

  bool subDetsOn_[4];

  ofstream htmlFile;

  DeadCellHists hbhists, hehists, hohists, hfhists;
  
  // Quality criteria for data integrity
  std::map<std::string, std::vector<QReport*> > dqmReportMapErr_;
  std::map<std::string, std::vector<QReport*> > dqmReportMapWarn_;
  std::map<std::string, std::vector<QReport*> > dqmReportMapOther_;
  std::map<std::string, std::string> dqmQtests_;

};

#endif
