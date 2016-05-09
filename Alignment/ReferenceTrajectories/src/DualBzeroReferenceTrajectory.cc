

#include "Alignment/ReferenceTrajectories/interface/DualBzeroReferenceTrajectory.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "Alignment/ReferenceTrajectories/interface/BzeroReferenceTrajectory.h"



DualBzeroReferenceTrajectory::DualBzeroReferenceTrajectory(const TrajectoryStateOnSurface& tsos,
                                                           const ConstRecHitContainer& forwardRecHits,
                                                           const ConstRecHitContainer& backwardRecHits,
                                                           const MagneticField* magField,
                                                           const reco::BeamSpot& beamSpot,
                                                           const ReferenceTrajectoryBase::Config& config) :
  DualReferenceTrajectory(tsos.localParameters().mixedFormatVector().kSize - 1,
                          numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits) - 1,
                          config),
    theMomentumEstimate(config.momentumEstimate)
{
    theValidityFlag = DualReferenceTrajectory::construct(tsos,
                                                         forwardRecHits,
                                                         backwardRecHits,
                                                         magField,
                                                         beamSpot);
}


ReferenceTrajectory*
DualBzeroReferenceTrajectory::construct(const TrajectoryStateOnSurface &referenceTsos, 
					const ConstRecHitContainer &recHits,
					double mass, MaterialEffects materialEffects,
					const PropagationDirection propDir,
					const MagneticField *magField,
					bool useBeamSpot,
					const reco::BeamSpot &beamSpot) const
{
  if (materialEffects >= breakPoints)  throw cms::Exception("BadConfig")
    << "[DualBzeroReferenceTrajectory::construct] Wrong MaterialEffects: " << materialEffects;
  
  ReferenceTrajectoryBase::Config config(materialEffects, propDir, mass, theMomentumEstimate);
  config.useBeamSpot = useBeamSpot;
  config.hitsAreReverse = false;
  return new BzeroReferenceTrajectory(referenceTsos, recHits, magField, beamSpot, config);
}


AlgebraicVector
DualBzeroReferenceTrajectory::extractParameters(const TrajectoryStateOnSurface &referenceTsos) const
{
  AlgebraicVector param = asHepVector<5>( referenceTsos.localParameters().mixedFormatVector() );
  return param.sub( 2, 5 );
}
