#ifndef REFERENCE_TRAJECTORY_H
#define REFERENCE_TRAJECTORY_H

/** Class which calculates all relevant properties of a reference
 *  trajectory, given the initial TrajectoryStateOnSurface, the RecHits
 *  of the corresponding reconstructed track and a mass hypothesis.
 *  By default, the mass is assumed to be the muon-mass, but can be
 *  changed via a SimpleConfigurable ( TkReferenceTrajectory:Mass ). No
 *  check whether the RecHits are valid is performed.
 *
 *  The covariance-matrix may include multiple-scattering or energy-
 *  loss effects or both. This can be defined in the constructor via
 *  the variable MaterialEffects materialEffects:
 *
 *  materialEffects =  none/multipleScattering/energyLoss/combined
 *
 *
 *  The coordinates of the measurements and the reference track are
 *  returned in the order ( here the transposed vector is shown ) 
 *
 *  m = ( x1, y1, x2, y2, ..., xN, yN ),
 *
 *  whereas the matrix, holding the derivatives of the measurements
 *  w.r.t. the initial track-parameters (p1,..., p5), is of the form
 *
 *  D = ( dx1/dp1, dx1/dp2, ..., dx1/dp5,
 *
 *        dy1/dp1, dy1/dp2, ..., dy1/dp5,
 *
 *        dx2/dp1, dx2/dp2, ..., dx2/dp5,
 *
 *        dy2/dp1, dy2/dp2, ..., dy2/dp5,
 *
 *           .        .             .
 *
 *           .        .             .
 *
 *        dxN/dp1, dxN/dp2, ..., dxN/dp5,
 *
 *        dyN/dp1, dyN/dp2, ..., dyN/dp5 )
 *
 * FIXME: comments about accessors and parameters...
 *
 *
 * Author     : Gero Flucke (based on code by Edmund Widl replacing ORCA's TkReferenceTrack)
 * date       : 2006/09/17
 * last update: $Date$
 * by         : $Author$
 *
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TrajectoryStateOnSurface;
class MagneticField;
class MaterialEffectsUpdator;
class BoundPlane;

class ReferenceTrajectory : public ReferenceTrajectoryBase
{

public:
  /**Constructor with Tsos at first hit and list of hits,
     the material effects to be considered and a particle mass
     FIXME: explain magField
   */
  ReferenceTrajectory(const TrajectoryStateOnSurface &referenceTsos,
		      const TransientTrackingRecHit::ConstRecHitContainer &recHits,
		      const MagneticField *magField, 
		      MaterialEffects materialEffects = combined, 
		      double mass = 0.10565836); // FIXME: muon mass

  virtual ~ReferenceTrajectory() {}

  virtual ReferenceTrajectory* clone() const // FIXME: verify copy constructor
    { return new ReferenceTrajectory(*this); }

private:
  /** internal method to calculate members
   */
  bool construct(const TrajectoryStateOnSurface &referenceTsos, 
		 const TransientTrackingRecHit::ConstRecHitContainer &recHits,
		 double mass, MaterialEffects materialEffects,
		 const MagneticField *magField);

  /** internal method to get apropriate updator
   */
  MaterialEffectsUpdator* createUpdator(MaterialEffects materialEffects, double mass) const;

  bool propagate(const TrajectoryStateOnSurface &previousTsos, const BoundPlane &surface,
		 const MagneticField *magField,
		 AlgebraicMatrix &newJacobian, TrajectoryStateOnSurface &newTsos) const;
  
  void fillMeasurementAndError(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
			       unsigned int iRow);
  void fillDerivatives(const AlgebraicMatrix &projection,
		       const AlgebraicMatrix &fullJacobian, unsigned int iRow);
  void fillTrajectoryPositions(const AlgebraicMatrix &projection, 
			       const AlgebraicVector &mixedLocalParams, 
			       unsigned int iRow);
  void addMaterialEffectsCov(const std::vector<AlgebraicMatrix> &allJacobians, 
			     const std::vector<AlgebraicMatrix> &allProjections,
			     const std::vector<AlgebraicSymMatrix> &allCurvChanges,
			     const std::vector<AlgebraicSymMatrix> &allDeltaParaCovs);
};

#endif
