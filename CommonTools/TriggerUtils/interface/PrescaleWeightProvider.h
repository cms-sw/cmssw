#ifndef CommonTools_TriggerUtils_PrescaleWeightProvider_h
#define CommonTools_TriggerUtils_PrescaleWeightProvider_h

// -*- C++ -*-
//
// Package:    CommonTools/TriggerUtils
// Class:      PrescaleWeightProvider
//
/*
  \class    PrescaleWeightProvider PrescaleWeightProvider.h "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"
  \brief

   This class takes a vector of HLT paths and returns a weight based on their
   HLT and L1 prescales. The weight is equal to the lowest combined (L1*HLT) prescale
   of the selected paths

  \author   Aram Avetisyan
*/

#include <memory>
#include <string>
#include <vector>
#include <type_traits>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

class L1GtTriggerMenuLite;

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
  class Run;
  class TriggerResults;
}  // namespace edm

class PrescaleWeightProvider {
  bool configured_;
  bool init_;
  std::unique_ptr<HLTPrescaleProvider> hltPrescaleProvider_;
  edm::Handle<L1GtTriggerMenuLite> triggerMenuLite_;

  std::vector<std::string> l1SeedPaths_;

  // configuration parameters
  unsigned verbosity_;               // optional (default: 0)
  edm::InputTag triggerResultsTag_;  // optional (default: "TriggerResults::HLT")
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::InputTag l1GtTriggerMenuLiteTag_;                            // optional (default: "l1GtTriggerMenuLite")
  edm::EDGetTokenT<L1GtTriggerMenuLite> l1GtTriggerMenuLiteToken_;  // optional (default: "l1GtTriggerMenuLite")
  std::vector<std::string> hltPaths_;

public:
  // The constructor must be called from the ED module's c'tor
  template <typename T>
  PrescaleWeightProvider(const edm::ParameterSet& config, edm::ConsumesCollector&& iC, T& module);

  template <typename T>
  PrescaleWeightProvider(const edm::ParameterSet& config, edm::ConsumesCollector& iC, T& module);

  ~PrescaleWeightProvider() {}

  // to be called from the ED module's beginRun() method
  void initRun(const edm::Run& run, const edm::EventSetup& setup);

  // to be called from the ED module's event loop method
  template <typename T = int>
  T prescaleWeight(const edm::Event& event, const edm::EventSetup& setup);

private:
  PrescaleWeightProvider(const edm::ParameterSet& config, edm::ConsumesCollector& iC);

  void parseL1Seeds(const std::string& l1Seeds);
};

template <typename T>
PrescaleWeightProvider::PrescaleWeightProvider(const edm::ParameterSet& config, edm::ConsumesCollector&& iC, T& module)
    : PrescaleWeightProvider(config, iC, module) {}

template <typename T>
PrescaleWeightProvider::PrescaleWeightProvider(const edm::ParameterSet& config, edm::ConsumesCollector& iC, T& module)
    : PrescaleWeightProvider(config, iC) {
  hltPrescaleProvider_ = std::make_unique<HLTPrescaleProvider>(config, iC, module);
}

template <typename T>
T PrescaleWeightProvider::prescaleWeight(const edm::Event& event, const edm::EventSetup& setup) {
  static_assert(std::is_same_v<T, double> or std::is_same_v<T, FractionalPrescale>,
                "\n\tPlease use prescaleWeight<double> or prescaleWeight<FractionalPrescale>"
                "\n\t(other types for HLT prescales are not supported anymore by PrescaleWeightProvider");
  if (!init_)
    return 1;

  // L1
  L1GtUtils const& l1GtUtils = hltPrescaleProvider_->l1GtUtils();

  // HLT
  HLTConfigProvider const& hltConfig = hltPrescaleProvider_->hltConfigProvider();

  edm::Handle<edm::TriggerResults> triggerResults;
  event.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid()) {
    if (verbosity_ > 0)
      edm::LogError("PrescaleWeightProvider::prescaleWeight")
          << "TriggerResults product not found for InputTag \"" << triggerResultsTag_.encode() << "\"";
    return 1;
  }

  const int SENTINEL(-1);
  int weight(SENTINEL);

  for (unsigned ui = 0; ui < hltPaths_.size(); ui++) {
    const std::string hltPath(hltPaths_.at(ui));
    unsigned hltIndex(hltConfig.triggerIndex(hltPath));
    if (hltIndex == hltConfig.size()) {
      if (verbosity_ > 0)
        edm::LogError("PrescaleWeightProvider::prescaleWeight") << "HLT path \"" << hltPath << "\" does not exist";
      continue;
    }
    if (!triggerResults->accept(hltIndex))
      continue;

    const std::vector<std::pair<bool, std::string> >& level1Seeds = hltConfig.hltL1GTSeeds(hltPath);
    if (level1Seeds.size() != 1) {
      if (verbosity_ > 0)
        edm::LogError("PrescaleWeightProvider::prescaleWeight")
            << "HLT path \"" << hltPath << "\" provides too many L1 seeds";
      return 1;
    }
    parseL1Seeds(level1Seeds.at(0).second);
    if (l1SeedPaths_.empty()) {
      if (verbosity_ > 0)
        edm::LogWarning("PrescaleWeightProvider::prescaleWeight")
            << "Failed to parse L1 seeds for HLT path \"" << hltPath << "\"";
      continue;
    }

    int l1Prescale(SENTINEL);
    for (unsigned uj = 0; uj < l1SeedPaths_.size(); uj++) {
      int l1TempPrescale(SENTINEL);
      int errorCode(0);
      if (level1Seeds.at(0).first) {  // technical triggers
        unsigned techBit(atoi(l1SeedPaths_.at(uj).c_str()));
        const std::string techName(*(triggerMenuLite_->gtTechTrigName(techBit, errorCode)));
        if (errorCode != 0)
          continue;
        if (!l1GtUtils.decision(event, techName, errorCode))
          continue;
        if (errorCode != 0)
          continue;
        l1TempPrescale = l1GtUtils.prescaleFactor(event, techName, errorCode);
        if (errorCode != 0)
          continue;
      } else {  // algorithmic triggers
        if (!l1GtUtils.decision(event, l1SeedPaths_.at(uj), errorCode))
          continue;
        if (errorCode != 0)
          continue;
        l1TempPrescale = l1GtUtils.prescaleFactor(event, l1SeedPaths_.at(uj), errorCode);
        if (errorCode != 0)
          continue;
      }
      if (l1TempPrescale > 0) {
        if (l1Prescale == SENTINEL || l1Prescale > l1TempPrescale)
          l1Prescale = l1TempPrescale;
      }
    }
    if (l1Prescale == SENTINEL) {
      if (verbosity_ > 0)
        edm::LogError("PrescaleWeightProvider::prescaleWeight")
            << "Unable to find the L1 prescale for HLT path \"" << hltPath << "\"";
      continue;
    }

    auto const prescale = l1Prescale * hltPrescaleProvider_->prescaleValue<T>(event, setup, hltPath);

    if (prescale > 0) {
      if (weight == SENTINEL || weight > prescale) {
        weight = prescale;
      }
    }
  }

  if (weight == SENTINEL) {
    if (verbosity_ > 0)
      edm::LogWarning("PrescaleWeightProvider::prescaleWeight")
          << "No valid weight for any requested HLT path, returning default weight of 1";
    return 1;
  }

  return weight;
}

#endif  // CommonTools_TriggerUtils_PrescaleWeightProvider_h
