// -*- C++ -*-
//
// Package:     HLTMuonOfflineAnalyzer
// Class:       HLTMuonOfflineAnalyzer
//

//
// Jason Slaunwhite and Jeff Klukas
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlotContainer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TPRegexp.h"

//////////////////////////////////////////////////////////////////////////////
//////// Define the interface ////////////////////////////////////////////////

class HLTMuonOfflineAnalyzer : public DQMEDAnalyzer {
public:
  explicit HLTMuonOfflineAnalyzer(const edm::ParameterSet &);

private:
  // Analyzer Methods
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  // Extra Methods
  std::vector<std::string> moduleLabels(const std::string &);

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;

  // Member Variables
  HLTMuonMatchAndPlotContainer plotterContainer_;
  HLTConfigProvider hltConfig_;
};

//////////////////////////////////////////////////////////////////////////////
//////// Namespaces, Typedefs, and Constants /////////////////////////////////

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;

using vstring = vector<string>;

//////////////////////////////////////////////////////////////////////////////
//////// Class Methods ///////////////////////////////////////////////////////

HLTMuonOfflineAnalyzer::HLTMuonOfflineAnalyzer(const ParameterSet &pset)
    : pset_(pset),
      hltProcessName_(pset.getParameter<string>("hltProcessName")),
      hltPathsToCheck_(pset.getParameter<vstring>("hltPathsToCheck")),
      plotterContainer_(consumesCollector(), pset) {}

vector<string> HLTMuonOfflineAnalyzer::moduleLabels(const string &path) {
  vector<string> modules = hltConfig_.moduleLabels(path);
  auto iter = modules.begin();
  while (iter != modules.end()) {
    if ((iter->find("Filtered") == string::npos) && (iter->find("hltL1s") == string::npos)) {
      iter = modules.erase(iter);
    } else if (iter->find("L1Filtered0") != string::npos)
      iter = modules.erase(iter);
    else
      ++iter;
  }
  //  iter = modules.begin();
  //  while (iter != modules.end()){
  //    if ((iter->find("Filtered0") == string::npos)){
  //      std::cout<< *iter << std::endl;
  //      iter = modules.erase(iter);
  //    }
  //    else ++iter;
  //  }

  return modules;
}

void HLTMuonOfflineAnalyzer::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Initialize hltConfig
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    LogError("HLTMuonVal") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  // Get the set of trigger paths we want to make plots for
  set<string> hltPaths;
  for (auto const &i : hltPathsToCheck_) {
    for (auto const &j : hltConfig_.triggerNames()) {
      if (j.find(i) != std::string::npos) {
        hltPaths.insert(j);
      }
    }
  }

  // Initialize the plotters
  set<string>::iterator iPath;
  vector<string>::const_iterator ilabel;
  for (iPath = hltPaths.begin(); iPath != hltPaths.end(); iPath++) {
    string path = *iPath;
    vector<string> labels = moduleLabels(path);
    bool isLastLabel = false;
    for (ilabel = labels.begin(); ilabel != labels.end(); ilabel++) {
      if (*ilabel == labels.back())
        isLastLabel = true;
      plotterContainer_.addPlotter(pset_, path, *ilabel, isLastLabel);
    }
  }
}

void HLTMuonOfflineAnalyzer::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &iRun,
                                            edm::EventSetup const &iSetup) {
  plotterContainer_.beginRun(iBooker, iRun, iSetup);
}

void HLTMuonOfflineAnalyzer::analyze(const Event &iEvent, const EventSetup &iSetup) {
  plotterContainer_.analyze(iEvent, iSetup);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonOfflineAnalyzer);
