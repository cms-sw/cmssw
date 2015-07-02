#ifndef FastSimulation_Tracking_TrackCandidateProducer_h
#define FastSimulation_Tracking_TrackCandidateProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"

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


class TrackingRecHit;

#include <vector>

class TrackCandidateProducer : public edm::stream::EDProducer <>
{
 public:
  
  explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
  virtual ~TrackCandidateProducer(){;}
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;
  
 private:

  unsigned int minNumberOfCrossedLayers;
  unsigned int maxNumberOfCrossedLayers;

  bool rejectOverlaps;
  bool splitHits;
  bool hitMasks_exist;
  bool hitCombinationMasks_exist;

 
  edm::InputTag simTracks_;

  // tokens & labels
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;
  edm::EDGetTokenT<FastTMRecHitCombinations> recHitToken;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
  edm::EDGetTokenT<std::vector<bool> > hitMasksToken;
  std::string propagatorLabel;
  
};

#endif
