#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"

ReferenceTrajectoryBase::ReferenceTrajectoryBase(unsigned int nPar, unsigned int nHits)
  : theValidityFlag(false), theTsosVec(),
    theMeasurements(nMeasPerHit * nHits), theMeasurementsCov(nMeasPerHit * nHits, 0),
    theTrajectoryPositions(nMeasPerHit * nHits), 
    theTrajectoryPositionCov(nMeasPerHit * nHits,0),
    theParameters(nPar), theDerivatives(nMeasPerHit * nHits, nPar, 0) 
{
  theTsosVec.reserve(nHits);
}
