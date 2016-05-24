#ifndef Alignment_ReferenceTrajectories_DualBzeroReferenceTrajectory_H
#define Alignment_ReferenceTrajectories_DualBzeroReferenceTrajectory_H

/**
 *  Class implementing the reference trajectory of a single particle
 *  in the absence of a magentic field, i.e. a straight line with 4
 *  parameters (no parameter related to the momentum). Here, not the parameters
 *  of the trajectory state of the first (or the last) hit are used,
 *  but at a surface in "the middle" of the track. By this the track is
 *  "split" into two parts (hence "dual"). The list of all hits
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
 *  A value for the momentum has to be provided to allow to calculate
 *  material effects.
 *
 * LIMITATIONS:
 *  So far all input hits have to be valid, but invalid hits
 *  would be needed to take into account the material effects in them...
 *
 */

#include "Alignment/ReferenceTrajectories/interface/DualReferenceTrajectory.h"

class BzeroReferenceTrajectory;


class DualBzeroReferenceTrajectory : public DualReferenceTrajectory
{

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

  DualBzeroReferenceTrajectory(const TrajectoryStateOnSurface& tsos,
                               const ConstRecHitContainer& forwardRecHits,
                               const ConstRecHitContainer& backwardRecHits,
                               const MagneticField* magField,
                               const reco::BeamSpot& beamSpot,
                               const ReferenceTrajectoryBase::Config& config);

  virtual ~DualBzeroReferenceTrajectory() {}

  virtual DualBzeroReferenceTrajectory* clone() const { return new DualBzeroReferenceTrajectory(*this); }

protected:

  virtual ReferenceTrajectory* construct(const TrajectoryStateOnSurface &referenceTsos, 
					 const ConstRecHitContainer &recHits,
					 double mass, MaterialEffects materialEffects,
					 const PropagationDirection propDir,
					 const MagneticField *magField,
					 bool useBeamSpot,
					 const reco::BeamSpot &beamSpot) const;

  virtual AlgebraicVector extractParameters(const TrajectoryStateOnSurface &referenceTsos) const;

  double theMomentumEstimate;

};

#endif
