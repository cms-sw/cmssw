
#include "Alignment/ReferenceTrajectories/interface/DualReferenceTrajectory.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"


DualReferenceTrajectory::DualReferenceTrajectory( const TrajectoryStateOnSurface &referenceTsos,
						  const ConstRecHitContainer &forwardRecHits,
						  const ConstRecHitContainer &backwardRecHits,
						  const MagneticField *magField,
						  MaterialEffects materialEffects,
						  PropagationDirection propDir,
						  double mass )
  : ReferenceTrajectoryBase( referenceTsos.localParameters().mixedFormatVector().kSize,
			     numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits) - 1 )
{
  theValidityFlag = this->construct( referenceTsos, forwardRecHits, backwardRecHits,
				     mass, materialEffects, propDir, magField );
}



DualReferenceTrajectory::DualReferenceTrajectory( unsigned int nPar, unsigned int nHits )
  : ReferenceTrajectoryBase( nPar, nHits )
{}


bool DualReferenceTrajectory::construct( const TrajectoryStateOnSurface &refTsos, 
					 const ConstRecHitContainer &forwardRecHits,
					 const ConstRecHitContainer &backwardRecHits,
					 double mass, MaterialEffects materialEffects,
					 const PropagationDirection propDir,
					 const MagneticField *magField)
{
  ReferenceTrajectoryBase* fwdTraj = construct(refTsos, forwardRecHits,
					       mass, materialEffects,
					       propDir, magField);

  ReferenceTrajectoryBase* bwdTraj = construct(refTsos, backwardRecHits,
					       mass, materialEffects,
					       oppositeDirection(propDir), magField);

  if ( !( fwdTraj->isValid() && bwdTraj->isValid() ) )
  {
    delete fwdTraj;
    delete bwdTraj;
    return false;
  }

  //
  // Combine both reference trajactories to a dual reference trajectory
  //

  const std::vector<TrajectoryStateOnSurface>& fwdTsosVec = fwdTraj->trajectoryStates();
  const std::vector<TrajectoryStateOnSurface>& bwdTsosVec = bwdTraj->trajectoryStates();
  theTsosVec.insert( theTsosVec.end(), fwdTsosVec.begin(), fwdTsosVec.end() );
  theTsosVec.insert( theTsosVec.end(), ++bwdTsosVec.begin(), bwdTsosVec.end() );

  ConstRecHitContainer fwdRecHits = fwdTraj->recHits();
  ConstRecHitContainer bwdRecHits = bwdTraj->recHits();
  theRecHits.insert( theRecHits.end(), fwdRecHits.begin(), fwdRecHits.end() );
  theRecHits.insert( theRecHits.end(), ++bwdRecHits.begin(), bwdRecHits.end() );

  theParameters = extractParameters( refTsos );

  unsigned int nParam = theParameters.num_row();
  unsigned int nFwdMeas = nMeasPerHit*fwdTraj->numberOfHits();
  unsigned int nBwdMeas = nMeasPerHit*bwdTraj->numberOfHits();

  theMeasurements.sub( 1, fwdTraj->measurements() );
  theMeasurements.sub( nFwdMeas+1, bwdTraj->measurements().sub( nMeasPerHit+1, nBwdMeas ) );

  theMeasurementsCov.sub( 1, fwdTraj->measurementErrors() );
  theMeasurementsCov.sub( nFwdMeas+1, bwdTraj->measurementErrors().sub( nMeasPerHit+1, nBwdMeas ) );

  theTrajectoryPositions.sub( 1, fwdTraj->trajectoryPositions() );
  theTrajectoryPositions.sub( nFwdMeas+1, bwdTraj->trajectoryPositions().sub( nMeasPerHit+1, nBwdMeas ) );

  theTrajectoryPositionCov.sub( 1, fwdTraj->trajectoryPositionErrors() );
  theTrajectoryPositionCov.sub( nFwdMeas+1, bwdTraj->trajectoryPositionErrors().sub( nMeasPerHit+1, nBwdMeas ) );

  theDerivatives.sub( 1, 1, fwdTraj->derivatives() );
  theDerivatives.sub( nFwdMeas+1, 1, bwdTraj->derivatives().sub( nMeasPerHit+1, nBwdMeas, 1, nParam ) );

  delete fwdTraj;
  delete bwdTraj;

  return true; 
}


ReferenceTrajectory*
DualReferenceTrajectory::construct(const TrajectoryStateOnSurface &referenceTsos, 
				   const ConstRecHitContainer &recHits,
				   double mass, MaterialEffects materialEffects,
				   const PropagationDirection propDir,
				   const MagneticField *magField) const
{
  return new ReferenceTrajectory(referenceTsos, recHits, false,
				 magField, materialEffects, propDir, mass);
}


AlgebraicVector
DualReferenceTrajectory::extractParameters(const TrajectoryStateOnSurface &referenceTsos) const
{
  return asHepVector<5>( referenceTsos.localParameters().mixedFormatVector() );
}
