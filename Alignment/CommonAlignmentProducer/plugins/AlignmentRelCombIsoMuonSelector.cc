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

class AlignmentRelCombIsoMuonSelector : public edm::global::EDFilter<> {
public:
  explicit AlignmentRelCombIsoMuonSelector(const edm::ParameterSet&);
  ~AlignmentRelCombIsoMuonSelector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const double relCombIsoCut_;
  const bool useTrackerOnlyIsolation_;  // New flag for tracker-only isolation
  const bool filterEvents_;
};

void AlignmentRelCombIsoMuonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("muons"))->setComment("Input muon collection");
  desc.add<double>("relCombIsoCut", 0.15)->setComment("cut on the relative combined isolation");
  desc.add<bool>("useTrackerOnlyIsolation", false)->setComment("use only tracker isolation");
  desc.add<bool>("filter", true);
  descriptions.addWithDefaultLabel(desc);
}

AlignmentRelCombIsoMuonSelector::AlignmentRelCombIsoMuonSelector(const edm::ParameterSet& iConfig)
    : muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      relCombIsoCut_(iConfig.getParameter<double>("relCombIsoCut")),
      useTrackerOnlyIsolation_(iConfig.getParameter<bool>("useTrackerOnlyIsolation")),
      filterEvents_(iConfig.getParameter<bool>("filter")) {
  produces<reco::MuonCollection>();
}

bool AlignmentRelCombIsoMuonSelector::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  auto selectedMuons = std::make_unique<reco::MuonCollection>();

  for (const auto& muon : *muons) {
    double relCombIso;
    if (useTrackerOnlyIsolation_) {
      // Tracker-only isolation
      relCombIso = muon.isolationR03().sumPt / muon.pt();
    } else {
      // Full combined isolation
      relCombIso = (muon.isolationR03().sumPt + muon.isolationR03().emEt + muon.isolationR03().hadEt) / muon.pt();
    }

    if (relCombIso < relCombIsoCut_) {
      selectedMuons->push_back(muon);
    }
  }

  const bool passEvent = !selectedMuons->empty();
  iEvent.put(std::move(selectedMuons));

  // Apply the filter flag logic
  return filterEvents_ ? passEvent : true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentRelCombIsoMuonSelector);
-- dummy change --
