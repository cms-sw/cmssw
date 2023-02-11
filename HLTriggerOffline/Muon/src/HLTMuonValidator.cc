// -*- C++ -*-
//
// Package:     HLTMuonValidator
// Class:       HLTMuonValidator
//
// Jason Slaunwhite and Jeff Klukas
//
#include <algorithm>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTriggerOffline/Muon/interface/HLTMuonPlotter.h"

#include "TPRegexp.h"

class HLTMuonValidator : public DQMEDAnalyzer {
public:
  explicit HLTMuonValidator(const edm::ParameterSet &);

private:
  // Analyzer Methods
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  // Extra Methods
  void fillLabels(std::string const &path,
                  std::vector<std::string> &moduleLabels,
                  std::vector<std::string> &stepLabels) const;
  std::string stepLabel(std::string const &moduleLabel) const;

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;

  // Member Variables
  std::vector<HLTMuonPlotter> analyzers_;
  HLTConfigProvider hltConfig_;

  edm::EDGetTokenT<trigger::TriggerEventWithRefs> const triggerEventToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> const genParticlesToken_;
  edm::EDGetTokenT<reco::MuonCollection> const recoMuonsToken_;

  HLTMuonPlotter::L1MuonMatcherAlgoForDQM const l1tMuonMatcherAlgo_;
};

HLTMuonValidator::HLTMuonValidator(const edm::ParameterSet &pset)
    : pset_(pset),
      hltProcessName_(pset.getParameter<std::string>("hltProcessName")),
      hltPathsToCheck_(pset.getParameter<std::vector<std::string>>("hltPathsToCheck")),
      triggerEventToken_(consumes(edm::InputTag("hltTriggerSummaryRAW"))),
      genParticlesToken_(consumes(pset.getParameter<std::string>("genParticleLabel"))),
      recoMuonsToken_(consumes(pset.getParameter<std::string>("recMuonLabel"))),
      l1tMuonMatcherAlgo_(pset, consumesCollector()) {}

void HLTMuonValidator::fillLabels(std::string const &path,
                                  std::vector<std::string> &moduleLabels,
                                  std::vector<std::string> &stepLabels) const {
  auto const &hltFilters = hltConfig_.saveTagsModules(path);

  moduleLabels.clear();
  moduleLabels.reserve(hltFilters.size());

  stepLabels.clear();
  stepLabels.reserve(hltFilters.size() + 1);

  for (auto const &module : hltFilters) {
    if (module.find("Filtered") == std::string::npos)
      continue;

    auto const step_label = stepLabel(module);
    if (step_label.empty() or std::find(stepLabels.begin(), stepLabels.end(), step_label) != stepLabels.end())
      continue;

    moduleLabels.emplace_back(module);
    stepLabels.emplace_back(step_label);
  }

  if (stepLabels.empty()) {
    return;
  }

  if (stepLabels[0] != "L1" and std::find(stepLabels.begin(), stepLabels.end(), "L1") != stepLabels.end()) {
    edm::LogWarning wrn("HLTMuonValidator");
    wrn << "Unsupported list of 'step' labels (the label 'L1' is present, but is not the first one): stepLabels=(";
    for (auto const &foo : stepLabels)
      wrn << " " << foo;
    wrn << " )";

    moduleLabels.clear();
    stepLabels.clear();
    return;
  }

  stepLabels.insert(stepLabels.begin(), "All");
}

std::string HLTMuonValidator::stepLabel(std::string const &module) const {
  if (module.find("IsoFiltered") != std::string::npos) {
    return (module.find("L3") != std::string::npos) ? "L3TkIso" : "L2Iso";
  } else if (module.find("pfecalIsoRhoFiltered") != std::string::npos) {
    if (module.find("L3") != std::string::npos)
      return "L3EcalIso";
    else if (module.find("TkFiltered") != std::string::npos)
      return "TkEcalIso";
  } else if (module.find("pfhcalIsoRhoFiltered") != std::string::npos) {
    if (module.find("L3") != std::string::npos)
      return "L3HcalIso";
    else if (module.find("TkFiltered") != std::string::npos)
      return "TkHcalIso";
  } else if (module.find("TkFiltered") != std::string::npos)
    return "Tk";
  else if (module.find("L3") != std::string::npos)
    return "L3";
  else if (module.find("L2") != std::string::npos)
    return "L2";
  else if (module.find("L1") != std::string::npos)
    return "L1";

  return "";
}

void HLTMuonValidator::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Initialize hltConfig
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTMuonVal") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  // Get the set of trigger paths we want to make plots for
  std::set<std::string> hltPaths;
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    TPRegexp pattern(hltPathsToCheck_[i]);
    for (size_t j = 0; j < hltConfig_.triggerNames().size(); j++)
      if (TString(hltConfig_.triggerNames()[j]).Contains(pattern))
        hltPaths.insert(hltConfig_.triggerNames()[j]);
  }

  // Initialize the analyzers
  analyzers_.clear();
  std::set<std::string>::iterator iPath;
  for (iPath = hltPaths.begin(); iPath != hltPaths.end(); iPath++) {
    const std::string &path = *iPath;
    std::string shortpath = path;
    if (path.rfind("_v") < path.length())
      shortpath = path.substr(0, path.rfind("_v"));

    std::vector<std::string> labels;
    std::vector<std::string> steps;
    fillLabels(path, labels, steps);

    if (!labels.empty() && !steps.empty()) {
      HLTMuonPlotter analyzer(
          pset_, shortpath, labels, steps, triggerEventToken_, genParticlesToken_, recoMuonsToken_, l1tMuonMatcherAlgo_);
      analyzers_.push_back(analyzer);
    }
  }
}

void HLTMuonValidator::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup) {
  // Call the beginRun (which books all the histograms)
  for (auto &analyzer : analyzers_) {
    analyzer.beginRun(iBooker, iRun, iSetup);
  }
}

void HLTMuonValidator::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  for (auto &analyzer : analyzers_) {
    analyzer.analyze(iEvent, iSetup);
  }
}

// define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonValidator);
