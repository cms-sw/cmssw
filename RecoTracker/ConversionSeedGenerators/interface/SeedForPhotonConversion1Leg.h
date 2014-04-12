#ifndef SeedForPhotonConversion1Leg_H
#define SeedForPhotonConversion1Leg_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"


#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class FreeTrajectoryState;

//
// this class need to be cleaned and optimized as those in RecoTracker/TkSeedGenerator
//
class SeedForPhotonConversion1Leg  {
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
  ~SeedForPhotonConversion1Leg(){}

  const TrajectorySeed * trajectorySeed( TrajectorySeedCollection & seedCollection,
						 const SeedingHitSet & hits,
						 const GlobalPoint & vertex,
						 const GlobalVector & vertexBounds,
						 float ptmin,
						 const edm::EventSetup& es,
						 float cotTheta,
						 std::stringstream& ss);

  
 protected:

  bool checkHit(
			const TrajectoryStateOnSurface &,
			const SeedingHitSet::ConstRecHitPointer &hit,
			const edm::EventSetup& es) const { return true; }

  GlobalTrajectoryParameters initialKinematic(
						      const SeedingHitSet & hits, 
						      const GlobalPoint & vertexPos, 
						      const edm::EventSetup& es,
						      const float cotTheta) const;
  
  CurvilinearTrajectoryError initialError(
						  const GlobalVector& vertexBounds, 
						  float ptMin,  
						  float sinTheta) const;
  
  const TrajectorySeed * buildSeed(
					   TrajectorySeedCollection & seedCollection,
					   const SeedingHitSet & hits,
					   const FreeTrajectoryState & fts,
					   const edm::EventSetup& es) const;

  SeedingHitSet::RecHitPointer refitHit( SeedingHitSet::ConstRecHitPointer hit, 
					 const TrajectoryStateOnSurface &state) const;
  
protected:
  std::string thePropagatorLabel;
  double theBOFFMomentum;

  // FIXME (well the whole class needs to be fixed!)      
  mutable  TkClonerImpl cloner;

  std::stringstream * pss;
  PrintRecoObjects po;
};
#endif 
