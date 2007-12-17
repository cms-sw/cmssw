#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

ReferenceTrajectoryBase::ReferenceTrajectoryBase(unsigned int nPar, unsigned int nHits)
  : theValidityFlag(false), theParamCovFlag(false),
    theNumberOfHits( nHits ),
    theTsosVec(), theRecHits(),
    theMeasurements(nMeasPerHit * nHits), theMeasurementsCov(nMeasPerHit * nHits, 0),
    theTrajectoryPositions(nMeasPerHit * nHits), 
    theTrajectoryPositionCov(nMeasPerHit * nHits,0),
    theParameters(nPar), theParameterCov(nPar, 0),
    theDerivatives(nMeasPerHit * nHits, nPar, 0) 
{
  theTsosVec.reserve(nHits);
  theRecHits.reserve(nHits);
}
