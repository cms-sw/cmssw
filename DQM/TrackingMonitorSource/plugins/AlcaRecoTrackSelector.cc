#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/AlcaRecoTrackSelector.h"

using namespace std;
using namespace edm;

AlcaRecoTrackSelector::AlcaRecoTrackSelector(const edm::ParameterSet& ps)
    : tracksTag_(ps.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      ptmin_(ps.getUntrackedParameter<double>("ptmin", 0.)),
      pmin_(ps.getUntrackedParameter<double>("pmin", 0.)),
      etamin_(ps.getUntrackedParameter<double>("etamin", -4.)),
      etamax_(ps.getUntrackedParameter<double>("etamax", 4.)),
      nhits_(ps.getUntrackedParameter<uint32_t>("nhits", 1)) {
  produces<reco::TrackCollection>("");
}

void AlcaRecoTrackSelector::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection());

  // Read Track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  if (tracks.isValid()) {
    for (auto const& trk : *tracks) {
      if (trk.pt() < ptmin_)
        continue;
      if (trk.p() < pmin_)
        continue;
      if (trk.eta() < etamin_)
        continue;
      if (trk.eta() > etamax_)
        continue;
      if (trk.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS) <= nhits_)
        continue;
      outputTColl->push_back(trk);
    }

  } else {
    edm::LogError("AlcaRecoTrackSelector") << "Error >> Failed to get AlcaRecoTrackSelector for label: " << tracksTag_;
  }

  iEvent.put(std::move(outputTColl));
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlcaRecoTrackSelector);
