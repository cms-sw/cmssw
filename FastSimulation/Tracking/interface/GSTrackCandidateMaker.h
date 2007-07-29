#ifndef FastSimulation_Tracking_GSTrackCandidateMaker_h
#define FastSimulation_Tracking_GSTrackCandidateMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <string>

class TransientInitialStateEstimator;
class MagneticField;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
class ParticlePropagator; 

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class GSTrackCandidateMaker : public edm::EDProducer
{
 public:

  explicit GSTrackCandidateMaker(const edm::ParameterSet& conf);
  
  virtual ~GSTrackCandidateMaker();
  
  virtual void beginJob (edm::EventSetup const & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  void stateOnDet(const TrajectoryStateOnSurface& ts,
		  unsigned int detid,
		  PTrajectoryStateOnDet& pts) const;
  
  bool compatibleWithVertex(GlobalPoint& gpos1, GlobalPoint& gpos2); 

 private:

  const MagneticField*  theMagField;
  const TrackerGeometry*  theGeometry;

  double pTMin;
  double maxD0;
  double maxZ0;
  unsigned minRecHits;
  std::string hitProducer;

  bool seedCleaning;
  bool rejectOverlaps;
  unsigned int seedType;
  double originRadius;
  double originHalfLength;
  double originpTMin;
  //PR-like hit rejection
  std::string fname;
  double chi2Cut;
  unsigned int minTkHits;

};

#endif
