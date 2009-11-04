#ifndef FastSimulation_Tracking_TrackCandidateProducer_h
#define FastSimulation_Tracking_TrackCandidateProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class TrackerGeometry;
class TrajectoryStateOnSurface;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace reco { 
  class Track;
}

class TrackerRecHit;

#include <vector>

class TrackCandidateProducer : public edm::EDProducer
{
 public:
  
  explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
  virtual ~TrackCandidateProducer();
  
  virtual void beginRun(edm::Run & run, const edm::EventSetup & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  int findId(const reco::Track& aTrack) const;

  void addSplitHits(const TrackerRecHit&, std::vector<TrackerRecHit>&); 

 private:

  const TrackerGeometry*  theGeometry;

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
};

#endif
