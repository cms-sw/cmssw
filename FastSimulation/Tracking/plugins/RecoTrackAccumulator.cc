/** \class RecoTrackAccumulator
 *
 * RecoTrackAccumulator accumulates generalTracks from the hard and the pileup events
 *
 * \author Andrea Giammanco
 *
 * \version   Mar 11 2013
 *
 ************************************************************/

#include <memory>
#include <vector>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class RecoTrackAccumulator : public DigiAccumulatorMixMod {
public:
  explicit RecoTrackAccumulator(const edm::ParameterSet& conf, edm::ProducesCollector, edm::ConsumesCollector& iC);
  ~RecoTrackAccumulator() override;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

private:
  template <class T>
  void accumulateEvent(const T& e, edm::EventSetup const& c, const edm::InputTag& label);

  std::unique_ptr<reco::TrackCollection> newTracks_;
  std::unique_ptr<reco::TrackExtraCollection> newTrackExtras_;
  std::unique_ptr<TrackingRecHitCollection> newHits_;
  std::unique_ptr<std::vector<Trajectory>> newTrajs_;
  std::unique_ptr<TrajTrackAssociationCollection> newTTAss_;

  reco::TrackRefProd rNewTracks_;
  reco::TrackExtraRefProd rNewTrackExtras_;
  TrackingRecHitRefProd rNewHits_;
  edm::RefProd<std::vector<Trajectory>> rNewTrajs_;
  edm::RefProd<TrajTrackAssociationCollection> rNewTTAss_;

  edm::InputTag signalTracksTag_;
  edm::InputTag pileUpTracksTag_;
  bool copyTrajectories_;
  std::string outputLabel_;
};

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector& iC)
    : signalTracksTag_(conf.getParameter<edm::InputTag>("signalTracks")),
      pileUpTracksTag_(conf.getParameter<edm::InputTag>("pileUpTracks")),
      copyTrajectories_(conf.getParameter<bool>("copyTrajectories")),
      outputLabel_(conf.getParameter<std::string>("outputLabel")) {
  producesCollector.produces<reco::TrackCollection>(outputLabel_);
  producesCollector.produces<TrackingRecHitCollection>(outputLabel_);
  producesCollector.produces<reco::TrackExtraCollection>(outputLabel_);

  iC.consumes<reco::TrackCollection>(signalTracksTag_);
  iC.consumes<TrackingRecHitCollection>(signalTracksTag_);
  iC.consumes<reco::TrackExtraCollection>(signalTracksTag_);

  if (copyTrajectories_) {
    producesCollector.produces<std::vector<Trajectory>>(outputLabel_);
    producesCollector.produces<TrajTrackAssociationCollection>(outputLabel_);

    iC.consumes<std::vector<Trajectory>>(signalTracksTag_);
    iC.consumes<TrajTrackAssociationCollection>(signalTracksTag_);
  }
}

RecoTrackAccumulator::~RecoTrackAccumulator() {}

void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
  newTracks_ = std::make_unique<reco::TrackCollection>();
  newHits_ = std::make_unique<TrackingRecHitCollection>();
  newTrackExtras_ = std::make_unique<reco::TrackExtraCollection>();

  // this is needed to get the ProductId of the TrackExtra and TrackingRecHit and Track collections
  rNewTracks_ = const_cast<edm::Event&>(e).getRefBeforePut<reco::TrackCollection>(outputLabel_);
  rNewTrackExtras_ = const_cast<edm::Event&>(e).getRefBeforePut<reco::TrackExtraCollection>(outputLabel_);
  rNewHits_ = const_cast<edm::Event&>(e).getRefBeforePut<TrackingRecHitCollection>(outputLabel_);

  if (copyTrajectories_) {
    newTrajs_ = std::make_unique<std::vector<Trajectory>>();
    newTTAss_ = std::make_unique<TrajTrackAssociationCollection>(
        const_cast<edm::Event&>(e).getRefBeforePut<std::vector<Trajectory>>(outputLabel_),
        const_cast<edm::Event&>(e).getRefBeforePut<reco::TrackCollection>(outputLabel_));

    rNewTrajs_ = const_cast<edm::Event&>(e).getRefBeforePut<std::vector<Trajectory>>(outputLabel_);
    rNewTTAss_ = const_cast<edm::Event&>(e).getRefBeforePut<TrajTrackAssociationCollection>(outputLabel_);
  }
}

void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  accumulateEvent(e, iSetup, signalTracksTag_);
}

void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e,
                                      edm::EventSetup const& iSetup,
                                      edm::StreamID const&) {
  if (e.bunchCrossing() == 0) {
    accumulateEvent(e, iSetup, pileUpTracksTag_);
  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  e.put(std::move(newTracks_), outputLabel_);
  e.put(std::move(newHits_), outputLabel_);
  e.put(std::move(newTrackExtras_), outputLabel_);
  if (copyTrajectories_) {
    e.put(std::move(newTrajs_), outputLabel_);
    e.put(std::move(newTTAss_), outputLabel_);
  }
}

template <class T>
void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup, const edm::InputTag& label) {
  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<TrackingRecHitCollection> hits;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  edm::Handle<std::vector<Trajectory>> trajs;
  edm::Handle<TrajTrackAssociationCollection> ttass;
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

  if (copyTrajectories_) {
    e.getByLabel(label, trajs);
    e.getByLabel(label, ttass);
    if (!trajs.isValid()) {
      throw cms::Exception("RecoTrackAccumulator")
          << "Failed to find trajectories with inputTag " << label << std::endl;
    }
    if (!ttass.isValid()) {
      throw cms::Exception("RecoTrackAccumulator")
          << "Failed to find trajectory-track association map with inputTag " << label << std::endl;
    }
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
    newTracks_->back().setExtra(reco::TrackExtraRef(rNewTrackExtras_, newTracks_->size() - 1));
    // rechits:
    // note: extra.recHit(i) does not work for pileup events
    // probably the Ref does not know its product id applies on a pileup event
    auto& newExtra = newTrackExtras_->back();
    auto const firstTrackIndex = newHits_->size();
    for (unsigned int i = 0; i < extra.recHitsSize(); i++) {
      newHits_->push_back((*hits)[extra.recHit(i).key()]);
    }
    newExtra.setHits(rNewHits_, firstTrackIndex, newHits_->size() - firstTrackIndex);
    newExtra.setTrajParams(extra.trajParams(), extra.chi2sX5());
    assert(newExtra.recHitsSize() == newExtra.trajParams().size());
    if (copyTrajectories_) {
      newTrajs_->emplace_back((*trajs)[t]);
      newTTAss_->insert(edm::Ref<std::vector<Trajectory>>(rNewTrajs_, newTrajs_->size() - 1),
                        reco::TrackRef(rNewTracks_, newTracks_->size() - 1));
    }
  }
}

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
DEFINE_DIGI_ACCUMULATOR(RecoTrackAccumulator);
