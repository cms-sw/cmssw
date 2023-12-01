// system include files
#include <memory>
#include <algorithm>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

// class declaration
class AlcaRecoTrackSelector : public edm::global::EDProducer<> {
public:
  explicit AlcaRecoTrackSelector(const edm::ParameterSet&);
  ~AlcaRecoTrackSelector() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

  const edm::InputTag tracksTag_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const double ptmin_;
  const double pmin_;
  const double etamin_;
  const double etamax_;
  const int nhits_;
};

void AlcaRecoTrackSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"));
  desc.addUntracked<double>("ptmin", 0.);
  desc.addUntracked<double>("pmin", 0.);
  desc.addUntracked<double>("etamin", -4.);
  desc.addUntracked<double>("etamax", 4.);
  desc.addUntracked<unsigned int>("nhits", 1);
  descriptions.addWithDefaultLabel(desc);
}

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
