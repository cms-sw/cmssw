#include "RecoMuon/MuonIdentification/plugins/MuonTrackExtraThinningProducer.h"
#include "DataFormats/MuonReco/interface/Muon.h"

MuonTrackExtraSelector::MuonTrackExtraSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc)
    : cut_(pset.getParameter<std::string>("cut")),
      muonToken_(cc.consumes<edm::View<reco::Muon> >(pset.getParameter<edm::InputTag>("muonTag"))),
      selector_(cut_) {}

void MuonTrackExtraSelector::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::string>("cut");
  desc.add<edm::InputTag>("muonTag");
}

void MuonTrackExtraSelector::preChooseRefs(edm::Handle<reco::TrackExtraCollection> trackExtras,
                                           edm::Event const& event,
                                           edm::EventSetup const& es) {
  auto muons = event.getHandle(muonToken_);

  for (const auto& muon : *muons) {
    if (!selector_(muon)) {
      continue;
    }
    addRef(muon.bestTrack()->extra());
  }
}
