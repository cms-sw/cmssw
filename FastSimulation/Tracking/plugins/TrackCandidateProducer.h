#ifndef FastSimulation_Tracking_TrackCandidateProducer_h
#define FastSimulation_Tracking_TrackCandidateProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

class TrackerGeometry;
class TrajectoryStateOnSurface;
class PropagatorWithMaterial;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace reco { 
  class Track;
}

class TrackerRecHit;
class TrackingRecHit;

#include <vector>

class TrackCandidateProducer : public edm::EDProducer
{
 public:
  
  explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
  virtual ~TrackCandidateProducer();
  
  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;
  
 private:

  int findId(const reco::Track& aTrack) const;

  void addSplitHits(const TrackerRecHit&, std::vector<TrackerRecHit>&); 
  bool isDuplicateCandidate(const TrackCandidateCollection& candidates, const TrackCandidate& newCand) const;
  bool sameLocalParameters(const TrackingRecHit* aH, const TrackingRecHit* bH) const;

 private:

  const TrackerGeometry*  theGeometry;
  const MagneticField*  theMagField;
  PropagatorWithMaterial* thePropagator;


  edm::InputTag seedProducer;
  edm::InputTag hitProducer;
  // edm::InputTag trackProducer;
  std::vector<edm::InputTag> trackProducers;
  
  unsigned int minNumberOfCrossedLayers;
  unsigned int maxNumberOfCrossedLayers;

  bool rejectOverlaps;
  bool splitHits;
  bool seedCleaning;
  bool keepFittedTracks;

  edm::InputTag simTracks_;
  double estimatorCut_;

  // tokens
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;
  edm::EDGetTokenT<SiTrackerGSMatchedRecHit2DCollection> recHitToken;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > trackTokens;
  std::vector<edm::EDGetTokenT<std::vector<Trajectory> > > trajectoryTokens;
  std::vector<edm::EDGetTokenT<TrajTrackAssociationCollection> >  assoMapTokens;
};

#endif
