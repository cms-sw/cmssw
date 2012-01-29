#ifndef SeedForPhotonConversion1Leg_H
#define SeedForPhotonConversion1Leg_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"
class FreeTrajectoryState;

class SeedForPhotonConversion1Leg : public SeedCreator {
public:
  static const int cotTheta_Max=99999;
  
  SeedForPhotonConversion1Leg( const edm::ParameterSet & cfg):
    thePropagatorLabel(cfg.getParameter<std::string>("propagator")),
    theBOFFMomentum(cfg.existsAs<double>("SeedMomentumForBOFF") ? cfg.getParameter<double>("SeedMomentumForBOFF") : 5.0)
      {}

  SeedForPhotonConversion1Leg( 
      const std::string & propagator = "PropagatorWithMaterial", double seedMomentumForBOFF = -5.0) 
   : thePropagatorLabel(propagator), theBOFFMomentum(seedMomentumForBOFF) { }

  //dtor
  virtual ~SeedForPhotonConversion1Leg(){}

  virtual const TrajectorySeed * trajectorySeed( TrajectorySeedCollection & seedCollection,
						 const SeedingHitSet & hits,
						 const GlobalPoint & vertex,
						 const GlobalVector & vertexBounds,
						 float ptmin,
						 const edm::EventSetup& es,
						 float cotTheta,
						 std::stringstream& ss);

  virtual const TrajectorySeed *trajectorySeed(
					       TrajectorySeedCollection & seedCollection,
					       const SeedingHitSet & hits,
					       const TrackingRegion & region,
					       const edm::EventSetup& es,
                                               const SeedComparitor *filter){ return 0;}
  
 protected:

  virtual bool checkHit(
			const TrajectoryStateOnSurface &,
			const TransientTrackingRecHit::ConstRecHitPointer &hit,
			const edm::EventSetup& es) const { return true; }

  virtual GlobalTrajectoryParameters initialKinematic(
						      const SeedingHitSet & hits, 
						      const GlobalPoint & vertexPos, 
						      const edm::EventSetup& es,
						      const float cotTheta) const;
  
  virtual CurvilinearTrajectoryError initialError(
						  const GlobalVector& vertexBounds, 
						  float ptMin,  
						  float sinTheta) const;
  
  virtual const TrajectorySeed * buildSeed(
					   TrajectorySeedCollection & seedCollection,
					   const SeedingHitSet & hits,
					   const FreeTrajectoryState & fts,
					   const edm::EventSetup& es) const;

  virtual TransientTrackingRecHit::RecHitPointer refitHit(
							  const TransientTrackingRecHit::ConstRecHitPointer &hit, 
							  const TrajectoryStateOnSurface &state) const;
  
protected:
  std::string thePropagatorLabel;
  double theBOFFMomentum;

  std::stringstream * pss;
  PrintRecoObjects po;
};
#endif 
