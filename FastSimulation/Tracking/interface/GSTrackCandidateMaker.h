#ifndef FastSimulation_Tracking_GSTrackCandidateMaker_h
#define FastSimulation_Tracking_GSTrackCandidateMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TransientInitialStateEstimator;
class MagneticField;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
class ParticlePropagator; 

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

  edm::ParameterSet conf_;
  
  const MagneticField*  theMagField;
  const TrackerGeometry*  theGeometry;

  double pTMin;
  double maxD0;
  double maxZ0;
  unsigned minRecHits;
  GlobalPoint gpos1,gpos2;  
};

#endif
