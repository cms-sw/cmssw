#ifndef SeedForPhotonConversion1Leg_H
#define SeedForPhotonConversion1Leg_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "PrintRecoObjects.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class FreeTrajectoryState;

//
// this class need to be cleaned and optimized as those in RecoTracker/TkSeedGenerator
//
class dso_hidden SeedForPhotonConversion1Leg {
public:
  static const int cotTheta_Max = 99999;

  SeedForPhotonConversion1Leg(const edm::ParameterSet& cfg)
      : thePropagatorLabel(cfg.getParameter<std::string>("propagator")),
        theBOFFMomentum(cfg.getParameter<double>("SeedMomentumForBOFF")),
        TTRHBuilder(cfg.getParameter<std::string>("TTRHBuilder")) {}

  //dtor
  ~SeedForPhotonConversion1Leg() {}

  const TrajectorySeed* trajectorySeed(TrajectorySeedCollection& seedCollection,
                                       const SeedingHitSet& hits,
                                       const GlobalPoint& vertex,
                                       const GlobalVector& vertexBounds,
                                       float ptmin,
                                       const edm::EventSetup& es,
                                       float cotTheta,
                                       std::stringstream& ss);

protected:
  bool checkHit(const TrajectoryStateOnSurface&,
                const SeedingHitSet::ConstRecHitPointer& hit,
                const edm::EventSetup& es) const {
    return true;
  }

  GlobalTrajectoryParameters initialKinematic(const SeedingHitSet& hits,
                                              const GlobalPoint& vertexPos,
                                              const edm::EventSetup& es,
                                              const float cotTheta) const;

  CurvilinearTrajectoryError initialError(const GlobalVector& vertexBounds, float ptMin, float sinTheta) const;

  const TrajectorySeed* buildSeed(TrajectorySeedCollection& seedCollection,
                                  const SeedingHitSet& hits,
                                  const FreeTrajectoryState& fts,
                                  const edm::EventSetup& es) const;

  SeedingHitSet::RecHitPointer refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                        const TrajectoryStateOnSurface& state,
                                        const TkClonerImpl& cloner) const;

protected:
  std::string thePropagatorLabel;
  double theBOFFMomentum;
  std::string TTRHBuilder;

  std::stringstream* pss;
  PrintRecoObjects po;
};
#endif
