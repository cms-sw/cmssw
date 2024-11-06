#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class AlignmentGoodIdMuonSelector : public edm::global::EDFilter<> {
public:
  explicit AlignmentGoodIdMuonSelector(const edm::ParameterSet&);
  ~AlignmentGoodIdMuonSelector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const double maxEta_;
  const double maxChi2_;
  const int minMuonHits_;
  const int minMatches_;
  const bool requireGlobal_;
  const bool requireTracker_;
  const bool filterEvents_;  // flag to control event filtering behavior

  // Secondary selection parameters (e.g., for Phase 2)
  const bool useSecondarySelection_;
  const double secondaryEtaLow_;
  const double secondaryEtaHigh_;
  const int secondaryMinMatches_;
  const bool requireTrackerForSecondary_;
};

void AlignmentGoodIdMuonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("muons"))->setComment("Input muon collection");
  desc.add<double>("maxEta", 2.5)->setComment("|eta| cut");
  desc.add<double>("maxChi2", 20.)->setComment("max chi2 of the global tags");
  desc.add<int>("minMuonHits", 0.)->setComment("minimum number of valid muon hits");
  desc.add<int>("minMatches", 1.)->setComment("minimum number of matches");
  desc.add<bool>("requireGlobal", true)->setComment("is global muons");
  desc.add<bool>("requireTracker", true)->setComment("is tracker muon");
  desc.add<bool>("useSecondarySelection", false)->setComment("secondary selection");
  desc.add<double>("secondaryEtaLow", 2.3)->setComment("min eta cut (secondary)");
  desc.add<double>("secondaryEtaHigh", 3.0)->setComment("max eta cut (secondary)");
  desc.add<int>("secondaryMinMatches", 0.)->setComment("minimum number of matches (secondary)");
  desc.add<bool>("secondaryRequireTracker", true)->setComment("is tracker muon (secondary)");
  desc.add<bool>("filter", true)->setComment("retain event only if non empty collection");
  descriptions.addWithDefaultLabel(desc);
}

AlignmentGoodIdMuonSelector::AlignmentGoodIdMuonSelector(const edm::ParameterSet& iConfig)
    : muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      maxEta_(iConfig.getParameter<double>("maxEta")),
      maxChi2_(iConfig.getParameter<double>("maxChi2")),
      minMuonHits_(iConfig.getParameter<int>("minMuonHits")),
      minMatches_(iConfig.getParameter<int>("minMatches")),
      requireGlobal_(iConfig.getParameter<bool>("requireGlobal")),
      requireTracker_(iConfig.getParameter<bool>("requireTracker")),
      filterEvents_(iConfig.getParameter<bool>("filter")),

      // Secondary selection
      useSecondarySelection_(iConfig.getParameter<bool>("useSecondarySelection")),
      secondaryEtaLow_(iConfig.getParameter<double>("secondaryEtaLow")),
      secondaryEtaHigh_(iConfig.getParameter<double>("secondaryEtaHigh")),
      secondaryMinMatches_(iConfig.getParameter<int>("secondaryMinMatches")),
      requireTrackerForSecondary_(iConfig.getParameter<bool>("secondaryRequireTracker")) {
  produces<reco::MuonCollection>();
}

bool AlignmentGoodIdMuonSelector::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  auto selectedMuons = std::make_unique<reco::MuonCollection>();

  for (const auto& muon : *muons) {
    bool passPrimarySelection = true;

    // Check if globalTrack() is valid before using it
    if (requireGlobal_) {
      if (!muon.isGlobalMuon() || muon.globalTrack().isNull()) {
        passPrimarySelection = false;
      } else {
        // Only access properties if the global track is valid
        if (muon.globalTrack()->hitPattern().numberOfValidMuonHits() <= minMuonHits_)
          passPrimarySelection = false;
        if (muon.globalTrack()->normalizedChi2() >= maxChi2_)
          passPrimarySelection = false;
      }
    }

    if (requireTracker_ && !muon.isTrackerMuon())
      passPrimarySelection = false;
    if (muon.numberOfMatches() <= minMatches_)
      passPrimarySelection = false;
    if (std::abs(muon.eta()) >= maxEta_)
      passPrimarySelection = false;

    bool passSecondarySelection = false;
    if (useSecondarySelection_) {
      if (std::abs(muon.eta()) > secondaryEtaLow_ && std::abs(muon.eta()) < secondaryEtaHigh_ &&
          muon.numberOfMatches() >= secondaryMinMatches_ && (!requireTrackerForSecondary_ || muon.isTrackerMuon())) {
        passSecondarySelection = true;
      }
    }

    if (passPrimarySelection || passSecondarySelection) {
      selectedMuons->push_back(muon);
    }
  }

  const bool passEvent = !selectedMuons->empty();
  iEvent.put(std::move(selectedMuons));

  // Decide if the event should pass based on filterEvents_ flag
  return filterEvents_ ? passEvent : true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentGoodIdMuonSelector);
