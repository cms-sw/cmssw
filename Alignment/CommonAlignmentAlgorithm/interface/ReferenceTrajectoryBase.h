#ifndef REFERENCE_TRAJECTORY_BASE_H
#define REFERENCE_TRAJECTORY_BASE_H

/**
 * Author     : Gero Flucke (based on code for ORCA by Edmund Widl)
 * date       : 2006/09/17
 * last update: $Date$
 * by         : $Author$
 *
 *
 *
 *
 */

#include "Geometry/Surface/interface/ReferenceCounted.h"

// AlgebraicVector, -Matrix and -SymMatrix
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>


class TrajectoryStateOnSurface; // FIXME: should not work, need include

class ReferenceTrajectoryBase : public ReferenceCounted
{

public:

  enum MaterialEffects { none, multipleScattering, energyLoss, combined };

  virtual ~ReferenceTrajectoryBase() {}

  bool isValid() { return theValidityFlag; }

  /** Returns the measurements in local coordinates.
   */
  const AlgebraicVector& measurements() const { return theMeasurements; }

  /** Returns the full covariance matrix of the measurements. ORCA-equivalent: covariance()
   */
  const AlgebraicSymMatrix& measurementErrors() const { return theMeasurementsCov; }

  /** Returns the local coordinates of the reference trajectory.
      ORCA-equivalent: referenceTrack()
   */
  const AlgebraicVector& trajectoryPositions() const { return theTrajectoryPositions; }

  /** Returns the covariance matrix of the reference trajectory.
      FIXME: what is that? Not in ORCA. @Edumnd: trajectoryErrors()
   */
  const AlgebraicSymMatrix& trajectoryPositionErrors() const { return theTrajectoryPositionCov; }

  /** Returns the derivatives of the local coordinates of the reference
   *  trajectory (i.e. trajectoryPositions) w.r.t. the initial 'track'-parameters.
   */
  const AlgebraicMatrix& derivatives() const { return theDerivatives; }

  /** Returns the set of 'track'-parameters.
   */
  const AlgebraicVector& parameters() const { return theParameters; }
  /** Returns the (input) hits
   */
/*   const TransientTrackingRecHit::ConstRecHitContainer& recHits() const { */
/*     return theRecHits;} */

  /** Returns the Tsos at the surfaces of the hits
   */
  const std::vector<TrajectoryStateOnSurface>& trajectoryStates() const { return theTsosVec; }

  virtual ReferenceTrajectoryBase* clone() const = 0;

protected:

  explicit ReferenceTrajectoryBase(unsigned int nPar = 0, unsigned int nHits = 0)
    : theValidityFlag(false), theTsosVec(nHits), /* theRecHits(nHits), */
    theMeasurements(nMeasPerHit * nHits), theMeasurementsCov(nMeasPerHit * nHits, 0),
    theTrajectoryPositions(nMeasPerHit * nHits), 
    theTrajectoryPositionCov(nMeasPerHit * nHits,0),
    theParameters(nPar), theDerivatives(nMeasPerHit * nHits, nPar, 0) {}
 

  bool theValidityFlag;

  std::vector<TrajectoryStateOnSurface> theTsosVec; // was theTsos
  //  TransientTrackingRecHit::ConstRecHitContainer theRecHits; // FIXME: would like to remove

  AlgebraicVector     theMeasurements;
  AlgebraicSymMatrix  theMeasurementsCov;

  AlgebraicVector     theTrajectoryPositions;   // was theTrajectory
  AlgebraicSymMatrix  theTrajectoryPositionCov; // was theTrajectoryCov; 

  AlgebraicVector     theParameters;
  AlgebraicMatrix     theDerivatives;

  static const unsigned int nMeasPerHit = 2;
};

#endif // REFERENCE_TRAJECTORY_BASE_H

