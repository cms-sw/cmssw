#ifndef Alignment_ReferenceTrajectories_BzeroReferenceTrajectory_h 
#define Alignment_ReferenceTrajectories_BzeroReferenceTrajectory_h 

/**
 *  Class implementing the reference trajectory of a single particle in
 *  the absence of a magnetic field (B=0, hence the name), i.e. a
 *  straight track with only 4 parameters (no curvature parameter).
 *
 *  Given the TrajectoryStateOnSurface at the first hit and the list of
 *  all hits the local measurements, derivatives etc. as described in
 *  (and accessed via) ReferenceTrajectoryBase are calculated.
 * 
 *  The covariance-matrix of the measurements may include effects of
 *  multiple-scattering or energy-loss effects or both. This can be
 *  defined in the constructor via the variable 'materialEffects
 *  (cf. ReferenceTrajectoryBase):
 *
 *  materialEffects =  none/multipleScattering/energyLoss/combined
 *
 *  Since no a-priori momentum estimate exists, but is needed for the
 *  calculation of material effects, a fixed value is used (e.g.
 *  representing the average momentum of the cosmics spectrum).
 *
 *  By default, the mass is assumed to be the muon-mass, but can be
 *  changed via a constructor argument.
 *
 * LIMITATIONS:
 *  So far all input hits have to be valid, but invalid hits
 *  would be needed to take into account the material effects in them...
 *
 */

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"

class BzeroReferenceTrajectory : public ReferenceTrajectory
{

public:
  /**Constructor with Tsos at first hit (in physical order) and list of hits 
     [if (hitsAreReverse) ==> order of hits is in opposite direction compared
     to the flight of particle, but note that BzeroReferenceTrajectory::recHits()
     returns the hits always in order of flight],
     the material effects to be considered and a particle mass and a momentum extimate,
     the magnetic field of the event is needed for propagations etc.
   */
  BzeroReferenceTrajectory(const TrajectoryStateOnSurface& tsos,
                           const TransientTrackingRecHit::ConstRecHitContainer& recHits,
                           const MagneticField *magField,
                           const reco::BeamSpot& beamSpot,
                           const ReferenceTrajectoryBase::Config& config);

  virtual ~BzeroReferenceTrajectory() {}

  virtual BzeroReferenceTrajectory* clone() const
    { return new BzeroReferenceTrajectory(*this); }

private:

  double theMomentumEstimate;
};

#endif
