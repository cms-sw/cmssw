#ifndef Alignment_ReferenceTrajectories_DualReferenceTrajectory_H
#define Alignment_ReferenceTrajectories_DualReferenceTrajectory_H

/**
 *  Class implementing the reference trajectory of a single charged
 *  particle, i.e. a helix with 5 parameters. Here, not the parameters
 *  of the trajectory state of the first (or the last) hit are used,
 *  but at a surface in "the middle" of the track. By this the track is
 *  "split" into two parts (hence "dual wield"). The list of all hits
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
 *  By default, the mass is assumed to be the muon-mass, but can be
 *  changed via a constructor argument.
 *
 * LIMITATIONS:
 *  So far all input hits have to be valid, but invalid hits
 *  would be needed to take into account the material effects in them...
 *
 */

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class ReferenceTrajectory;
namespace reco { class BeamSpot;}

class DualReferenceTrajectory : public ReferenceTrajectoryBase
{

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

  DualReferenceTrajectory(const TrajectoryStateOnSurface& tsos,
                          const ConstRecHitContainer& forwardRecHits,
                          const ConstRecHitContainer& backwardRecHits,
                          const MagneticField* magField,
                          const reco::BeamSpot& beamSpot,
                          const ReferenceTrajectoryBase::Config& config);

  virtual ~DualReferenceTrajectory() {}

  virtual DualReferenceTrajectory* clone() const { return new DualReferenceTrajectory(*this); }

protected:

  DualReferenceTrajectory(unsigned int nPar, unsigned int nHits,
                          const ReferenceTrajectoryBase::Config& config);

  /** internal method to calculate members
   */
  virtual bool construct(const TrajectoryStateOnSurface &referenceTsos, 
			 const ConstRecHitContainer &forwardRecHits,
			 const ConstRecHitContainer &backwardRecHits,
			 const MagneticField *magField,
			 const reco::BeamSpot &beamSpot);

  virtual ReferenceTrajectory* construct(const TrajectoryStateOnSurface &referenceTsos, 
					 const ConstRecHitContainer &recHits,
					 const MagneticField *magField,
					 const reco::BeamSpot &beamSpot,
					 const bool revertDirection = false) const;

  virtual AlgebraicVector extractParameters(const TrajectoryStateOnSurface &referenceTsos) const;

  inline const PropagationDirection oppositeDirection( const PropagationDirection propDir ) const
  { return ( propDir == anyDirection ) ? anyDirection : ( ( propDir == alongMomentum ) ? oppositeToMomentum : alongMomentum ); }

private:
  const double mass_;
  const MaterialEffects materialEffects_;
  const PropagationDirection propDir_;
  const bool useBeamSpot_;
};

#endif
