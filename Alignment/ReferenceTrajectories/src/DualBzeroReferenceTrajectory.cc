

#include "Alignment/ReferenceTrajectories/interface/DualBzeroReferenceTrajectory.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "Alignment/ReferenceTrajectories/interface/BzeroReferenceTrajectory.h"


DualBzeroReferenceTrajectory::DualBzeroReferenceTrajectory( const TrajectoryStateOnSurface &referenceTsos,
							    const ConstRecHitContainer &forwardRecHits,
							    const ConstRecHitContainer &backwardRecHits,
							    const MagneticField *magField,
							    MaterialEffects materialEffects,
							    PropagationDirection propDir,
							    double mass,
							    double momentumEstimate, 
							    bool useBeamSpot,
							    const reco::BeamSpot &beamSpot)
  : DualReferenceTrajectory(referenceTsos.localParameters().mixedFormatVector().kSize - 1,
			    numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits) - 1),
    theMomentumEstimate(momentumEstimate)
{
    theValidityFlag = DualReferenceTrajectory::construct(referenceTsos,
							 forwardRecHits,
							 backwardRecHits,
							 mass, materialEffects,
							 propDir, magField,
							 useBeamSpot, beamSpot);
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
  
  return new BzeroReferenceTrajectory(referenceTsos, recHits,
				      false, magField,
				      materialEffects, propDir,
				      mass, theMomentumEstimate, useBeamSpot, beamSpot);
}


AlgebraicVector
DualBzeroReferenceTrajectory::extractParameters(const TrajectoryStateOnSurface &referenceTsos) const
{
  AlgebraicVector param = asHepVector<5>( referenceTsos.localParameters().mixedFormatVector() );
  return param.sub( 2, 5 );
}
