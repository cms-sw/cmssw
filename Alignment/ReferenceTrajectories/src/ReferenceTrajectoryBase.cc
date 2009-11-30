#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

ReferenceTrajectoryBase::ReferenceTrajectoryBase(unsigned int nPar, unsigned int nHits, 
						 unsigned int nMsPar, unsigned int nMsMeas)
  : theValidityFlag(false), theParamCovFlag(false),
    theNumberOfHits( nHits ), theNumberOfPars( nPar ), 
    theNumberOfMsMeas( nMsMeas ), theNumberOfMsPars( nMsPar ),
    theTsosVec(), theRecHits(),
    theMeasurements(nMeasPerHit * nHits + nMsMeas), 
    theMeasurementsCov(nMeasPerHit * nHits + nMsMeas, 0),
    theTrajectoryPositions(nMeasPerHit * nHits), 
    theTrajectoryPositionCov(nMeasPerHit * nHits, 0),
    theParameters(nPar), 
    theParameterCov(nPar, 0),
    theDerivatives(nMeasPerHit * nHits + nMsMeas, nPar + nMsPar, 0),
    theInnerTrajectoryToCurvilinear( 5, 5, 0 ),
    theInnerLocalToTrajectory( 5, 5, 0 )    
{
  theTsosVec.reserve(nHits);
  theRecHits.reserve(nHits);
}


unsigned int
ReferenceTrajectoryBase::numberOfUsedRecHits( const TransientTrackingRecHit::ConstRecHitContainer &recHits ) const
{
  unsigned int nUsedHits = 0;
  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itHit;
  for ( itHit = recHits.begin(); itHit != recHits.end(); ++itHit ) if ( useRecHit( *itHit ) ) ++nUsedHits;
  return nUsedHits;
}


bool
ReferenceTrajectoryBase::useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const
{
  return hitPtr->isValid();
}
