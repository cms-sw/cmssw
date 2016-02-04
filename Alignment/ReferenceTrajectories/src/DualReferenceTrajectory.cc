
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
						  double mass, 
						  bool useBeamSpot,
						  const reco::BeamSpot &beamSpot)
  : ReferenceTrajectoryBase(referenceTsos.localParameters().mixedFormatVector().kSize,
			    numberOfUsedRecHits(forwardRecHits) + numberOfUsedRecHits(backwardRecHits) - 1,
			    0, 0)
{
  theValidityFlag = this->construct( referenceTsos, forwardRecHits, backwardRecHits,
				     mass, materialEffects, propDir, magField, useBeamSpot, beamSpot );
}



DualReferenceTrajectory::DualReferenceTrajectory(unsigned int nPar, unsigned int nHits)
  : ReferenceTrajectoryBase(nPar, nHits, 0, 0)
{}


bool DualReferenceTrajectory::construct( const TrajectoryStateOnSurface &refTsos, 
					 const ConstRecHitContainer &forwardRecHits,
					 const ConstRecHitContainer &backwardRecHits,
					 double mass, MaterialEffects materialEffects,
					 const PropagationDirection propDir,
					 const MagneticField *magField,
					 bool useBeamSpot,
					 const reco::BeamSpot &beamSpot)
{
  if (materialEffects >= breakPoints)  throw cms::Exception("BadConfig")
    << "[DualReferenceTrajectory::construct] Wrong MaterialEffects: " << materialEffects;
    
  ReferenceTrajectoryBase* fwdTraj = construct(refTsos, forwardRecHits,
					       mass, materialEffects,
					       propDir, magField, useBeamSpot, beamSpot);

  ReferenceTrajectoryBase* bwdTraj = construct(refTsos, backwardRecHits,
					       mass, materialEffects,
					       oppositeDirection(propDir), magField, false, beamSpot);

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

  const ConstRecHitContainer &fwdRecHits = fwdTraj->recHits();
  const ConstRecHitContainer &bwdRecHits = bwdTraj->recHits();
  theRecHits.insert( theRecHits.end(), fwdRecHits.begin(), fwdRecHits.end() );
  theRecHits.insert( theRecHits.end(), ++bwdRecHits.begin(), bwdRecHits.end() );

  theParameters = extractParameters( refTsos );
  
  unsigned int nParam   = theNumberOfPars;
  unsigned int nFwdMeas = fwdTraj->numberOfHitMeas();
  unsigned int nBwdMeas = bwdTraj->numberOfHitMeas();
  unsigned int nFwdBP   = fwdTraj->numberOfMsMeas();
  unsigned int nBwdBP   = bwdTraj->numberOfMsMeas();
  unsigned int nMeas    = nFwdMeas+nBwdMeas-nMeasPerHit; 
       
  theMeasurements.sub( 1, fwdTraj->measurements().sub( 1, nFwdMeas ) );
  theMeasurements.sub( nFwdMeas+1, bwdTraj->measurements().sub( nMeasPerHit+1, nBwdMeas ) );
    
  theMeasurementsCov.sub( 1, fwdTraj->measurementErrors().sub( 1, nFwdMeas ) );
  theMeasurementsCov.sub( nFwdMeas+1, bwdTraj->measurementErrors().sub( nMeasPerHit+1, nBwdMeas ) );
  
  theTrajectoryPositions.sub( 1, fwdTraj->trajectoryPositions() );
  theTrajectoryPositions.sub( nFwdMeas+1, bwdTraj->trajectoryPositions().sub( nMeasPerHit+1, nBwdMeas ) );

  theTrajectoryPositionCov.sub( 1, fwdTraj->trajectoryPositionErrors() );
  theTrajectoryPositionCov.sub( nFwdMeas+1, bwdTraj->trajectoryPositionErrors().sub( nMeasPerHit+1, nBwdMeas ) );

  theDerivatives.sub( 1, 1, fwdTraj->derivatives().sub( 1, nFwdMeas, 1, nParam ) );
  theDerivatives.sub( nFwdMeas+1, 1, bwdTraj->derivatives().sub( nMeasPerHit+1, nBwdMeas, 1, nParam ) );
  
// for the break points 
// DUAL with break points makes no sense: (MS) correlations between the two parts are lost ! 
  if (nFwdBP>0 )
  {
    theMeasurements.sub( nMeas+1, fwdTraj->measurements().sub( nFwdMeas+1, nFwdMeas+nFwdBP ) );
    theMeasurementsCov.sub( nMeas+1, fwdTraj->measurementErrors().sub( nFwdMeas+1, nFwdMeas+nFwdBP ) );  
    theDerivatives.sub( 1, nParam+1, fwdTraj->derivatives().sub( 1, nFwdMeas, nParam+1, nParam+nFwdBP ) );
    theDerivatives.sub( nMeas+1, nParam+1, fwdTraj->derivatives().sub( nFwdMeas+1, nFwdMeas+nFwdBP, nParam+1, nParam+nFwdBP ) );
  }  
  if (nBwdBP>0 )
  {
    theMeasurements.sub( nMeas+nFwdBP+1, bwdTraj->measurements().sub( nBwdMeas+1, nBwdMeas+nBwdBP ) );  
    theMeasurementsCov.sub( nMeas+nFwdBP+1, bwdTraj->measurementErrors().sub( nBwdMeas+1, nBwdMeas+nBwdBP ) );  
    theDerivatives.sub( nFwdMeas+1, nParam+nFwdBP+1, bwdTraj->derivatives().sub( nMeasPerHit+1, nBwdMeas, nParam+1, nParam+nBwdBP ) );
    theDerivatives.sub( nMeas+nFwdBP+1, nParam+nFwdBP+1, bwdTraj->derivatives().sub( nBwdMeas+1, nBwdMeas+nBwdBP, nParam+1, nParam+nBwdBP ) );
  }
    
  delete fwdTraj;
  delete bwdTraj;

  return true; 
}


ReferenceTrajectory*
DualReferenceTrajectory::construct(const TrajectoryStateOnSurface &referenceTsos, 
				   const ConstRecHitContainer &recHits,
				   double mass, MaterialEffects materialEffects,
				   const PropagationDirection propDir,
				   const MagneticField *magField,
				   bool useBeamSpot,
				   const reco::BeamSpot &beamSpot) const
{
  if (materialEffects >= breakPoints)  throw cms::Exception("BadConfig")
    << "[DualReferenceTrajectory::construct] Wrong MaterialEffects: " << materialEffects;
  
  return new ReferenceTrajectory(referenceTsos, recHits, false,
				 magField, materialEffects, propDir, mass,
				 useBeamSpot, beamSpot);
}


AlgebraicVector
DualReferenceTrajectory::extractParameters(const TrajectoryStateOnSurface &referenceTsos) const
{
  return asHepVector<5>( referenceTsos.localParameters().mixedFormatVector() );
}
