#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

ReferenceTrajectoryBase::ReferenceTrajectoryBase(unsigned int nPar, unsigned int nHits, unsigned int nBreakPoints)
  : theValidityFlag(false), theParamCovFlag(false),
    theNumberOfHits( nHits ), theNumberOfPars( nPar ), 
    theNumberOfBreakPoints( nBreakPoints ), //theNumberOfHitMeas( nMeasPerHit * nHits ),
    theTsosVec(), theRecHits(),
    theMeasurements(nMeasPerHit * nHits + nBreakPoints), 
    theMeasurementsCov(nMeasPerHit * nHits + nBreakPoints, 0),
    theTrajectoryPositions(nMeasPerHit * nHits), 
    theTrajectoryPositionCov(nMeasPerHit * nHits, 0),
    theParameters(nPar), 
    theParameterCov(nPar, 0),
    theDerivatives(nMeasPerHit * nHits + nBreakPoints, nPar + nBreakPoints, 0)
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
