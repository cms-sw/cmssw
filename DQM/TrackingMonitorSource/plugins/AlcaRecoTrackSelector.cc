#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TLorentzVector.h"
//#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/AlcaRecoTrackSelector.h"

using namespace std;
using namespace edm;

AlcaRecoTrackSelector::AlcaRecoTrackSelector(const edm::ParameterSet& ps)
    : parameters_(ps),
      tracksTag_(parameters_.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      ptmin_(parameters_.getUntrackedParameter<double>("ptmin", 0.)),
      pmin_(parameters_.getUntrackedParameter<double>("pmin", 0.)),
      etamin_(parameters_.getUntrackedParameter<double>("etamin", -4.)),
      etamax_(parameters_.getUntrackedParameter<double>("etamax", 4.)),
      nhits_(parameters_.getUntrackedParameter<uint32_t>("nhits", 1)) {
  produces<reco::TrackCollection>("");
}

void AlcaRecoTrackSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection());

  // Read Track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  if (tracks.isValid()) {
    for (auto const& trk : *tracks) {
      //      std::cout << "alcareco track pt : " << trk.pt() << "  |  eta : " << trk.eta() << "  |  nhits : " << trk.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS) << std::endl;
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
