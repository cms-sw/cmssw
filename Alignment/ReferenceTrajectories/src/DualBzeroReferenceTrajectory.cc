
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
							    double momentumEstimate )
  : DualReferenceTrajectory( referenceTsos.localParameters().mixedFormatVector().kSize - 1,
			     numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits) - 1,  (materialEffects == breakPoints) ? 2*(numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits))-4 : 0 ),
    theMomentumEstimate( momentumEstimate )
{
    theValidityFlag = DualReferenceTrajectory::construct( referenceTsos,
							forwardRecHits,
							backwardRecHits,
							mass, materialEffects,
							propDir, magField );
}


ReferenceTrajectory*
DualBzeroReferenceTrajectory::construct(const TrajectoryStateOnSurface &referenceTsos, 
				   const ConstRecHitContainer &recHits,
				   double mass, MaterialEffects materialEffects,
				   const PropagationDirection propDir,
				   const MagneticField *magField) const
{
  return new BzeroReferenceTrajectory(referenceTsos, recHits,
				      false, magField,
				      materialEffects, propDir,
				      mass, theMomentumEstimate);
}


AlgebraicVector
DualBzeroReferenceTrajectory::extractParameters(const TrajectoryStateOnSurface &referenceTsos) const
{
  AlgebraicVector param = asHepVector<5>( referenceTsos.localParameters().mixedFormatVector() );
  return param.sub( 2, 5 );
}
