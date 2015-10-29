#ifndef FastSimulation_Tracking_TrackCandidateProducer_h
#define FastSimulation_Tracking_TrackCandidateProducer_h

// framework stuff
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

// data formats
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"

// specific to this module
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "FastSimulation/Tracking/interface/FastTrackerRecHitSplitter.h"

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
  bool hitMasks_exists;
 
  FastTrackerRecHitSplitter hitSplitter;

  // tokens & labels
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;
  edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
  edm::EDGetTokenT<std::vector<bool> > hitMasksToken;
  std::string propagatorLabel;
  
};

#endif
