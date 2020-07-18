#include "RecoMuon/MuonIdentification/plugins/MuonTrackExtraThinningProducer.h"
#include "DataFormats/MuonReco/interface/Muon.h"

MuonTrackExtraSelector::MuonTrackExtraSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc)
    : cut_(pset.getParameter<std::string>("cut")),
      slimTrajParams_(pset.getParameter<bool>("slimTrajParams")),
      slimResiduals_(pset.getParameter<bool>("slimResiduals")),
      slimFinalState_(pset.getParameter<bool>("slimFinalState")),
      muonToken_(cc.consumes<edm::View<reco::Muon> >(pset.getParameter<edm::InputTag>("muonTag"))),
      selector_(cut_) {}

void MuonTrackExtraSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::string>("cut", "");
  desc.add<bool>("slimTrajParams", false);
  desc.add<bool>("slimResiduals", false);
  desc.add<bool>("slimFinalState", false);
  desc.add<edm::InputTag>("muonTag", edm::InputTag("muons"));
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

void MuonTrackExtraSelector::modify(reco::TrackExtra& trackExtra) {
  if (slimTrajParams_) {
    trackExtra.setTrajParams(reco::TrackExtraBase::TrajParams(), reco::TrackExtraBase::Chi2sFive());
  }
  if (slimResiduals_) {
    trackExtra.setResiduals(reco::TrackResiduals());
  }
  if (slimFinalState_) {
    if (trackExtra.seedDirection() == alongMomentum) {
      trackExtra.clearOuter();
    } else if (trackExtra.seedDirection() == oppositeToMomentum) {
      trackExtra.clearInner();
    }
  }
}
