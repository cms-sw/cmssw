#include "FastSimulation/Tracking/plugins/RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector& iC)
    : signalTracksTag(conf.getParameter<edm::InputTag>("signalTracks")),
      pileUpTracksTag(conf.getParameter<edm::InputTag>("pileUpTracks")),
      outputLabel(conf.getParameter<std::string>("outputLabel")) {
  producesCollector.produces<reco::TrackCollection>(outputLabel);
  producesCollector.produces<TrackingRecHitCollection>(outputLabel);
  producesCollector.produces<reco::TrackExtraCollection>(outputLabel);

  iC.consumes<reco::TrackCollection>(signalTracksTag);
  iC.consumes<TrackingRecHitCollection>(signalTracksTag);
  iC.consumes<reco::TrackExtraCollection>(signalTracksTag);
}

RecoTrackAccumulator::~RecoTrackAccumulator() {}

void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
  newTracks_ = std::unique_ptr<reco::TrackCollection>(new reco::TrackCollection);
  newHits_ = std::unique_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
  newTrackExtras_ = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);

  // this is needed to get the ProductId of the TrackExtra and TrackingRecHit and Track collections
  rNewTracks = const_cast<edm::Event&>(e).getRefBeforePut<reco::TrackCollection>(outputLabel);
  rNewTrackExtras = const_cast<edm::Event&>(e).getRefBeforePut<reco::TrackExtraCollection>(outputLabel);
  rNewHits = const_cast<edm::Event&>(e).getRefBeforePut<TrackingRecHitCollection>(outputLabel);
}

void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  accumulateEvent(e, iSetup, signalTracksTag);
}

void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e,
                                      edm::EventSetup const& iSetup,
                                      edm::StreamID const&) {
  if (e.bunchCrossing() == 0) {
    accumulateEvent(e, iSetup, pileUpTracksTag);
  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  e.put(std::move(newTracks_), outputLabel);
  e.put(std::move(newHits_), outputLabel);
  e.put(std::move(newTrackExtras_), outputLabel);
}

template <class T>
void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup, const edm::InputTag& label) {
  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<TrackingRecHitCollection> hits;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  e.getByLabel(label, tracks);
  e.getByLabel(label, hits);
  e.getByLabel(label, trackExtras);

  if (!tracks.isValid()) {
    throw cms::Exception("RecoTrackAccumulator")
        << "Failed to find track collections with inputTag " << label << std::endl;
  }
  if (!hits.isValid()) {
    throw cms::Exception("RecoTrackAccumulator")
        << "Failed to find hit collections with inputTag " << label << std::endl;
  }
  if (!trackExtras.isValid()) {
    throw cms::Exception("RecoTrackAccumulator")
        << "Failed to find trackExtra collections with inputTag " << label << std::endl;
  }

  for (size_t t = 0; t < tracks->size(); ++t) {
    const reco::Track& track = (*tracks)[t];
    newTracks_->push_back(track);
    // track extras:
    auto const& extra = trackExtras->at(track.extra().key());
    newTrackExtras_->emplace_back(extra.outerPosition(),
                                  extra.outerMomentum(),
                                  extra.outerOk(),
                                  extra.innerPosition(),
                                  extra.innerMomentum(),
                                  extra.innerOk(),
                                  extra.outerStateCovariance(),
                                  extra.outerDetId(),
                                  extra.innerStateCovariance(),
                                  extra.innerDetId(),
                                  extra.seedDirection(),
                                  //If TrajectorySeeds are needed, then their list must be gotten from the
                                  // secondary event directly and looked up similarly to TrackExtras.
                                  //We can't use a default constructed RefToBase due to a bug in RefToBase
                                  // which causes an seg fault when calling isAvailable on a default constructed one.
                                  edm::RefToBase<TrajectorySeed>{edm::Ref<std::vector<TrajectorySeed>>{}});
    newTracks_->back().setExtra(reco::TrackExtraRef(rNewTrackExtras, newTracks_->size() - 1));
    // rechits:
    // note: extra.recHit(i) does not work for pileup events
    // probably the Ref does not know its product id applies on a pileup event
    auto& newExtra = newTrackExtras_->back();
    auto const firstTrackIndex = newHits_->size();
    for (unsigned int i = 0; i < extra.recHitsSize(); i++) {
      newHits_->push_back((*hits)[extra.recHit(i).key()]);
    }
    newExtra.setHits(rNewHits, firstTrackIndex, newHits_->size() - firstTrackIndex);
    newExtra.setTrajParams(extra.trajParams(), extra.chi2sX5());
    assert(newExtra.recHitsSize() == newExtra.trajParams().size());
  }
}

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
DEFINE_DIGI_ACCUMULATOR(RecoTrackAccumulator);
