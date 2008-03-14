#ifndef Alignment_ReferenceTrajectories_ReferenceTrajectoryBase_H
#define Alignment_ReferenceTrajectories_ReferenceTrajectoryBase_H

/**
 * Author     : Gero Flucke (based on code for ORCA by Edmund Widl)
 * date       : 2006/09/17
 * last update: $Date: 2007/12/20 17:28:56 $
 * by         : $Author: ewidl $
 *
 * Base class for reference 'trajectories' of single- or multiparticles
 * stated.
 * Inheriting classes have to calculate all relevant quantities accessed
 * through member functions of this base class:
 *
 * The local measured x/y coordinates on all crossed detectors as vector:
 *
 * m = (x1, y1, x2, y2, ..., xN, yN) [transposed vector shown]
 *
 * their covariance matrix (possibly containing correlations between hits
 * due to material effects which are not taken into account by the ideal
 * trajectory parametrisation, cf. below),
 *
 * similarly the local x/y cordinates of the reference trajectory with covariance,
 *
 * the parameters of the (ideal) 'trajectory' 
 * (e.g. 5 values for a single track or 9 for a two-track-state with vertex constraint),
 *
 * the derivatives of the local coordinates of the reference trajectory
 * with respect to the initial 'trajectory'-parameters, 
 * e.g. for n parameters 'p' and N hits with 'x/y' coordinates:
 * 
 *  D = ( dx1/dp1, dx1/dp2, ..., dx1/dpn,
 *
 *        dy1/dp1, dy1/dp2, ..., dy1/dpn,
 *
 *        dx2/dp1, dx2/dp2, ..., dx2/dpn,
 *
 *        dy2/dp1, dy2/dp2, ..., dy2/dpn,
 *
 *           .        .             .
 *
 *           .        .             .
 *
 *        dxN/dp1, dxN/dp2, ..., dxN/dpn,
 *
 *        dyN/dp1, dyN/dp2, ..., dyN/dpn )
 *
 * and finally the TrajectoryStateOnSurface's of the reference trajectory.
 * 
 * Take care to check validity of the calculation (isValid()) before using the results.
 *
 */

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

// for AlgebraicVector, -Matrix and -SymMatrix:
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

#include <vector>

class ReferenceTrajectoryBase : public ReferenceCounted
{

public:

  typedef ReferenceCountingPointer<ReferenceTrajectoryBase> ReferenceTrajectoryPtr;

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
   */
  const AlgebraicSymMatrix& trajectoryPositionErrors() const { return theTrajectoryPositionCov; }

  /** Returns the derivatives of the local coordinates of the reference
   *  trajectory (i.e. trajectoryPositions) w.r.t. the initial 'trajectory'-parameters.
   */
  const AlgebraicMatrix& derivatives() const { return theDerivatives; }

  /** Returns the set of 'track'-parameters.
   */
  const AlgebraicVector& parameters() const { return theParameters; }

  /** Returns true if the covariance matrix of the 'track'-parameters is set.
   */
  inline bool parameterErrorsAvailable() const { return theParamCovFlag; }

  /** Set the covariance matrix of the 'track'-parameters.
   */
  inline void setParameterErrors( const AlgebraicSymMatrix& error ) { theParameterCov = error; theParamCovFlag = true; }

  /** Returns the covariance matrix of the 'track'-parameters.
   */
  inline const AlgebraicSymMatrix& parameterErrors() const { return theParameterCov; }

  /** Returns the Tsos at the surfaces of the hits, parallel to recHits()
   */
  const std::vector<TrajectoryStateOnSurface>& trajectoryStates() const { return theTsosVec; }

  /** Returns the TransientTrackingRecHits (as ConstRecHitPointer's), 
      order might depend on concrete implementation of inheriting class
   */
  const TransientTrackingRecHit::ConstRecHitContainer& recHits() const { return theRecHits; }

  inline int numberOfHits() const { return theNumberOfHits; }

  virtual ReferenceTrajectoryBase* clone() const = 0;

protected:

  explicit ReferenceTrajectoryBase(unsigned int nPar = 0, unsigned int nHits = 0);

  bool theValidityFlag;
  bool theParamCovFlag;

  unsigned int theNumberOfHits;

  std::vector<TrajectoryStateOnSurface> theTsosVec;
  TransientTrackingRecHit::ConstRecHitContainer theRecHits;

  AlgebraicVector     theMeasurements;
  AlgebraicSymMatrix  theMeasurementsCov;

  AlgebraicVector     theTrajectoryPositions;
  AlgebraicSymMatrix  theTrajectoryPositionCov;

  AlgebraicVector     theParameters;
  AlgebraicSymMatrix  theParameterCov;

  AlgebraicMatrix     theDerivatives;

  static const unsigned int nMeasPerHit = 2;
};

#endif // REFERENCE_TRAJECTORY_BASE_H

