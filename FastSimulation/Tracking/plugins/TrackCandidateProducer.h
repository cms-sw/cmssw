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

class TrackCandidateProducer : public edm::EDProducer
{
 public:
  
  explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
  virtual ~TrackCandidateProducer();
  
  virtual void beginJob (edm::EventSetup const & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  int findId(const reco::Track& aTrack) const;

 private:

  const TrackerGeometry*  theGeometry;

  edm::InputTag seedProducer;
  edm::InputTag hitProducer;
  edm::InputTag trackProducer;
  
  unsigned int minNumberOfCrossedLayers;
  unsigned int maxNumberOfCrossedLayers;

  bool rejectOverlaps;
  bool seedCleaning;

};

#endif
