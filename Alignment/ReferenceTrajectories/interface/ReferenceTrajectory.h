#ifndef Alignment_ReferenceTrajectories_ReferenceTrajectory_H
#define Alignment_ReferenceTrajectories_ReferenceTrajectory_H

/**
 * Author     : Gero Flucke (based on code by Edmund Widl replacing ORCA's TkReferenceTrack)
 * date       : 2006/09/17
 * last update: $Date: 2012/06/20 12:07:28 $
 * by         : $Author: flucke $
 *
 *  Class implementing the reference trajectory of a single charged
 *  particle, i.e. a helix with 5 parameters. Given the
 *  TrajectoryStateOnSurface at the first hit and the list of all hits
 *  the local measurements, derivatives etc. as described in (and
 *  accessed via) ReferenceTrajectoryBase are calculated.
 * 
 *  The covariance-matrix of the measurements may include effects of
 *  multiple-scattering or energy-loss effects or both. This can be
 *  defined in the constructor via the variable 'materialEffects
 *  (cf. ReferenceTrajectoryBase):
 *
 *  materialEffects =  none/multipleScattering/energyLoss/combined
 *
 *  Correct multiple scattering treatment, even if ignoring the off-diagonal
 *  elements of the covariance matrix, can be achieved with 
 *  (cf. documentaion of ReferenceTrajectoryBase):
 *
 *  materialEffects = BrokenLines[Coarse]/BrokenLinesFine/BreakPoints
 *
 *  By default, the mass is assumed to be the muon-mass, but can be
 *  changed via a constructor argument.
 *
 * LIMITATIONS:
 *  Only broken lines and break points parameterisations take into account 
 *  material effects of invalid hits.
 *
 */

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

class TrajectoryStateOnSurface;
class MagneticField;
class MaterialEffectsUpdator;
class Plane;
class BeamSpotTransientTrackingRecHit;

namespace reco { class BeamSpot; }

class ReferenceTrajectory : public ReferenceTrajectoryBase
{

public:

  typedef SurfaceSideDefinition::SurfaceSide SurfaceSide;

  /**Constructor with Tsos at first hit (in physical order) and list of hits 
     [if (hitsAreReverse) ==> order of hits is in opposite direction compared
     to the flight of particle, but note that ReferenceTrajectory::recHits()
     returns the hits always in order of flight],
     the material effects to be considered and a particle mass,
     the magnetic field and beamSpot of the event are needed for propagations etc.
   */
  ReferenceTrajectory(const TrajectoryStateOnSurface &referenceTsos,
		      const TransientTrackingRecHit::ConstRecHitContainer &recHits,
		      bool hitsAreReverse,
		      const MagneticField *magField,
		      MaterialEffects materialEffects,
		      PropagationDirection propDir,
		      double mass,
		      bool useBeamSpot,
		      const reco::BeamSpot &beamSpot);
  virtual ~ReferenceTrajectory() {}

  virtual ReferenceTrajectory* clone() const { return new ReferenceTrajectory(*this); }

protected:

  ReferenceTrajectory(unsigned int nPar, unsigned int nHits, MaterialEffects materialEffects);

  /** internal method to calculate members
   */
  virtual bool construct(const TrajectoryStateOnSurface &referenceTsos, 
			 const TransientTrackingRecHit::ConstRecHitContainer &recHits,
			 double mass, MaterialEffects materialEffects,
			 const PropagationDirection propDir, const MagneticField *magField,
			 bool useBeamSpot,
			 const reco::BeamSpot &beamSpot);

  /** internal method to get apropriate updator
   */
  MaterialEffectsUpdator* createUpdator(MaterialEffects materialEffects, double mass) const;

  /** internal method to calculate jacobian
   */
  virtual bool propagate(const Plane &previousSurface, const TrajectoryStateOnSurface &previousTsos,
			 const Plane &newSurface, TrajectoryStateOnSurface &newTsos, AlgebraicMatrix &newJacobian, 
			 AlgebraicMatrix &newCurvlinJacobian, double &nextStep,
			 const PropagationDirection propDir, const MagneticField *magField) const;
  
  /** internal method to fill measurement and error matrix for hit iRow/2
   */
  virtual void fillMeasurementAndError(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
				       unsigned int iRow,
				       const TrajectoryStateOnSurface &updatedTsos);

  /** internal method to fill derivatives for hit iRow/2
   */
  virtual void fillDerivatives(const AlgebraicMatrix &projection,
			       const AlgebraicMatrix &fullJacobian, unsigned int iRow);

  /** internal method to fill the trajectory positions for hit iRow/2
   */
  virtual void fillTrajectoryPositions(const AlgebraicMatrix &projection, 
				       const AlgebraicVector &mixedLocalParams, 
				       unsigned int iRow);

  /** internal method to add material effects to measurments covariance matrix
   */
  virtual bool addMaterialEffectsCov(const std::vector<AlgebraicMatrix> &allJacobians, 
				     const std::vector<AlgebraicMatrix> &allProjections,
				     const std::vector<AlgebraicSymMatrix> &allCurvChanges,
				     const std::vector<AlgebraicSymMatrix> &allDeltaParaCovs);
				     
  /** internal method to add material effects using break points
   */
  virtual bool addMaterialEffectsBp (const std::vector<AlgebraicMatrix> &allJacobians, 
				     const std::vector<AlgebraicMatrix> &allProjections,
				     const std::vector<AlgebraicSymMatrix> &allCurvChanges,
				     const std::vector<AlgebraicSymMatrix> &allDeltaParaCovs,
				     const std::vector<AlgebraicMatrix> &allLocalToCurv);
				     
  /** internal methods to add material effects using broken lines (fine version)
   */
  virtual bool addMaterialEffectsBrl(const std::vector<AlgebraicMatrix> &allJacobians, 
				     const std::vector<AlgebraicMatrix> &allProjections,
				     const std::vector<AlgebraicSymMatrix> &allCurvChanges,
				     const std::vector<AlgebraicSymMatrix> &allDeltaParaCovs,
				     const std::vector<AlgebraicMatrix> &allLocalToCurv,
				     const GlobalTrajectoryParameters &gtp);
  /** internal methods to add material effects using broken lines (coarse version)
   */
  virtual bool addMaterialEffectsBrl(const std::vector<AlgebraicMatrix> &allProjections,
				     const std::vector<AlgebraicSymMatrix> &allDeltaParaCovs,
				     const std::vector<AlgebraicMatrix> &allLocalToCurv,
				     const std::vector<double> &allSteps,
				     const GlobalTrajectoryParameters &gtp,   
				     const double minStep = 1.0);
          
  /// Don't care for propagation direction 'anyDirection' - in that case the material effects
  /// are anyway not updated ...
  inline SurfaceSide surfaceSide(const PropagationDirection dir) const
  {
    return ( dir == alongMomentum ) ?
      SurfaceSideDefinition::beforeSurface :
      SurfaceSideDefinition::afterSurface;
  }

  /** first (generic) helper to get the projection matrix
   */
  AlgebraicMatrix
    getHitProjectionMatrix(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;

  /** second helper (templated on the dimension) to get the projection matrix
   */
  template <unsigned int N>
    AlgebraicMatrix
    getHitProjectionMatrixT(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;
};

#endif
