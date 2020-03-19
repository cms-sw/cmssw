#ifndef RecoMuon_CosmicMuonProducer_GlobalCosmicMuonTrajectoryBuilder_H
#define RecoMuon_CosmicMuonProducer_GlobalCosmicMuonTrajectoryBuilder_H

/** \file GlobalCosmicMuonTrajectoryBuilder
 *  class to build combined trajectory from cosmic tracks in tk and mu
 *
 *  \author Chang Liu  -  Purdue University
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class Trajectory;
class TrajectoryMeasurement;
class CosmicMuonUtilities;

class GlobalCosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder {
public:
  typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

  /// Constructor
  GlobalCosmicMuonTrajectoryBuilder(const edm::ParameterSet&,
                                    const MuonServiceProxy* service,
                                    edm::ConsumesCollector& iC);

  /// Destructor
  ~GlobalCosmicMuonTrajectoryBuilder() override;

  /// dummy implementation, unused in this class
  TrajectoryContainer trajectories(const TrajectorySeed&) override { return TrajectoryContainer{}; }

  const Propagator* propagator() const { return &*theService->propagator(thePropagatorName); }

  /// choose tk Track and build combined trajectories
  CandidateContainer trajectories(const TrackCand&) override;

  /// check if tk and muon Tracks are matched
  std::vector<TrackCand> match(const TrackCand&, const edm::Handle<reco::TrackCollection>&);

  void setEvent(const edm::Event&) override;

private:
  void sortHits(ConstRecHitContainer&, ConstRecHitContainer&, ConstRecHitContainer&);

  ConstRecHitContainer getTransientRecHits(const reco::Track&) const;

  CosmicMuonSmoother* smoother() const { return theSmoother; }

  const CosmicMuonUtilities* utilities() const { return smoother()->utilities(); }

  bool isTraversing(const reco::Track& tk) const;

  const MuonServiceProxy* theService;

  CosmicMuonSmoother* theSmoother;

  GlobalMuonTrackMatcher* theTrackMatcher;

  std::string thePropagatorName;
  edm::EDGetTokenT<reco::TrackCollection> theTkTrackToken;

  std::string theTrackerRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;

  std::string theMuonRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;

  edm::Handle<reco::TrackCollection> theTrackerTracks;

  bool tkTrajsAvailable;

  const std::vector<Trajectory>* allTrackerTrajs;

  std::string category_;
};
#endif
