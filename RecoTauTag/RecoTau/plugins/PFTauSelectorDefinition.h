#ifndef RecoTauTag_RecoTau_PFTauSelectorDefinition
#define RecoTauTag_RecoTau_PFTauSelectorDefinition

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <iostream>

struct PFTauSelectorDefinition {
  typedef reco::PFTauCollection collection;
  typedef edm::Handle<collection> HandleToCollection;
  typedef std::vector<const reco::PFTau*> container;
  typedef container::const_iterator const_iterator;

  struct DiscCutPair {
    edm::Handle<reco::PFTauDiscriminator> handle;
    edm::EDGetTokenT<reco::PFTauDiscriminator> inputToken;
    double cut;
  };
  struct DiscContainerCutPair {
    edm::Handle<reco::TauDiscriminatorContainer> handle;
    edm::EDGetTokenT<reco::TauDiscriminatorContainer> inputToken;
    std::vector<std::string> rawLabels;
    std::vector<std::pair<int, double>> rawCuts;
    std::vector<std::string> wpLabels;
    std::vector<int> wpCuts;
  };
  typedef std::vector<DiscCutPair> DiscCutPairVec;
  typedef std::vector<DiscContainerCutPair> DiscContainerCutPairVec;

  PFTauSelectorDefinition(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) {
    auto const& discriminators = cfg.getParameter<std::vector<edm::ParameterSet>>("discriminators");
    auto const& discriminatorContainers = cfg.getParameter<std::vector<edm::ParameterSet>>("discriminatorContainers");
    // Build each of our cuts
    for (auto const& pset : discriminators) {
      DiscCutPair newCut;
      newCut.inputToken = iC.consumes<reco::PFTauDiscriminator>(pset.getParameter<edm::InputTag>("discriminator"));
      newCut.cut = pset.getParameter<double>("selectionCut");
      discriminators_.push_back(newCut);
    }
    for (auto const& pset : discriminatorContainers) {
      DiscContainerCutPair newCut;
      newCut.inputToken =
          iC.consumes<reco::TauDiscriminatorContainer>(pset.getParameter<edm::InputTag>("discriminator"));
      auto const& rawLabels = pset.getParameter<std::vector<std::string>>("rawValues");
      auto const& rawCutValues = pset.getParameter<std::vector<double>>("selectionCuts");
      if (rawLabels.size() != rawCutValues.size()) {
        throw cms::Exception("PFTauSelectorBadHandle")
            << "unequal number of TauIDContainer raw value indices and cut values given to PFTauSelector.";
      }
      for (size_t i = 0; i < rawLabels.size(); i++) {
        newCut.rawCuts.push_back(std::pair<int, double>(-99, rawCutValues[i]));
        newCut.rawLabels.push_back(rawLabels[i]);
      }
      newCut.wpLabels = pset.getParameter<std::vector<std::string>>("workingPoints");
      newCut.wpCuts.resize(newCut.wpLabels.size());
      discriminatorContainers_.push_back(newCut);
    }

    // Build a string cut if desired
    if (cfg.exists("cut")) {
      cut_.reset(new StringCutObjectSelector<reco::PFTau>(cfg.getParameter<std::string>("cut")));
    }
  }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }

  void select(const HandleToCollection& hc, const edm::Event& e, const edm::EventSetup& s) {
    selected_.clear();

    if (!hc.isValid()) {
      throw cms::Exception("PFTauSelectorBadHandle")
          << "an invalid PFTau handle with ProductID" << hc.id() << " passed to PFTauSelector.";
    }

    // Load each discriminator
    for (auto& disc : discriminators_) {
      e.getByToken(disc.inputToken, disc.handle);
    }
    for (auto& disc : discriminatorContainers_) {
      e.getByToken(disc.inputToken, disc.handle);
    }
    // Retrieve ID container indices if config history changes, in particular for the first event.
    if (phID_ != e.processHistoryID()) {
      phID_ = e.processHistoryID();
      for (auto& disc : discriminatorContainers_) {
        auto const& psetsFromProvenance = edm::parameterSet(disc.handle.provenance()->stable(), e.processHistory());
        // find raw value indices
        if (psetsFromProvenance.exists("rawValues")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<std::string>>("rawValues");
          for (size_t i = 0; i < disc.rawLabels.size(); ++i) {
            bool found = false;
            for (size_t j = 0; j < idlist.size(); ++j) {
              if (disc.rawLabels[i] == idlist[j]) {
                found = true;
                disc.rawCuts[i].first = j;
              }
            }
            if (!found)
              throw cms::Exception("Configuration")
                  << "PFTauSelector: Requested working point '" << disc.rawLabels[i] << "' not found!\n";
          }
        } else if (psetsFromProvenance.exists("IDdefinitions")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<edm::ParameterSet>>("IDdefinitions");
          for (size_t i = 0; i < disc.rawLabels.size(); ++i) {
            bool found = false;
            for (size_t j = 0; j < idlist.size(); ++j) {
              if (disc.rawLabels[i] == idlist[j].getParameter<std::string>("IDname")) {
                found = true;
                disc.rawCuts[i].first = j;
              }
            }
            if (!found)
              throw cms::Exception("Configuration")
                  << "PFTauSelector: Requested working point '" << disc.rawLabels[i] << "' not found!\n";
          }
        } else
          throw cms::Exception("Configuration") << "PFTauSelector: No suitable ID list found in provenace config!\n";
        // find working point indices
        if (psetsFromProvenance.exists("workingPoints")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<std::string>>("workingPoints");
          for (size_t i = 0; i < disc.wpLabels.size(); ++i) {
            bool found = false;
            for (size_t j = 0; j < idlist.size(); ++j) {
              if (disc.wpLabels[i] == idlist[j]) {
                found = true;
                disc.wpCuts[i] = j;
              }
            }
            if (!found)
              throw cms::Exception("Configuration")
                  << "PFTauSelector: Requested working point '" << disc.wpLabels[i] << "' not found!\n";
          }
        } else if (psetsFromProvenance.exists("IDWPdefinitions")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
          for (size_t i = 0; i < disc.wpLabels.size(); ++i) {
            bool found = false;
            for (size_t j = 0; j < idlist.size(); ++j) {
              if (disc.wpLabels[i] == idlist[j].getParameter<std::string>("IDname")) {
                found = true;
                disc.wpCuts[i] = j;
              }
            }
            if (!found)
              throw cms::Exception("Configuration")
                  << "PFTauSelector: Requested working point '" << disc.wpLabels[i] << "' not found!\n";
          }
        } else
          throw cms::Exception("Configuration") << "PFTauSelector: No suitable ID list found in provenace config!\n";
      }
    }

    const size_t nTaus = hc->size();
    for (size_t iTau = 0; iTau < nTaus; ++iTau) {
      bool passed = true;
      reco::PFTauRef tau(hc, iTau);
      // Check if it passed all the discrimiantors
      for (auto const& disc : discriminators_) {
        // Check this discriminator passes
        if (!((*disc.handle)[tau] > disc.cut)) {
          passed = false;
          break;
        }
      }
      if (passed) {  // if passed so far, check other discriminators
        for (auto const& disc : discriminatorContainers_) {
          for (auto const& rawCut : disc.rawCuts) {
            if (!((*disc.handle)[tau].rawValues.at(rawCut.first) > rawCut.second)) {
              passed = false;
              break;
            }
          }
          if (!passed)
            break;
          for (auto const& wpCut : disc.wpCuts) {
            if (!((*disc.handle)[tau].workingPoints.at(wpCut))) {
              passed = false;
              break;
            }
          }
          if (!passed)
            break;
        }
      }

      if (passed && cut_.get()) {
        passed = (*cut_)(*tau);
      }

      if (passed)
        selected_.push_back(tau.get());
    }
  }  // end select()

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  DiscCutPairVec discriminators_;
  DiscContainerCutPairVec discriminatorContainers_;
  edm::ProcessHistoryID phID_;

  std::unique_ptr<StringCutObjectSelector<reco::PFTau>> cut_;
};

#endif
