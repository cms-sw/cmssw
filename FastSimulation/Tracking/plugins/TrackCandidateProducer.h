#ifndef FastSimulation_Tracking_TrackCandidateProducer_h
#define FastSimulation_Tracking_TrackCandidateProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class TransientInitialStateEstimator;
class MagneticField;
class MagneticFieldMap;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
class ParticlePropagator; 

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrackCandidateProducer : public edm::EDProducer
{
 public:
  
  explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
  virtual ~TrackCandidateProducer();
  
  virtual void beginJob (edm::EventSetup const & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  const TrackerGeometry*  theGeometry;

  edm::InputTag seedProducer;
  edm::InputTag hitProducer;

  bool rejectOverlaps;

};

#endif
