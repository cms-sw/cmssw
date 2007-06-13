#ifndef Alignment_ReferenceTrajectories_CosmicsReferenceTrajectory_h 
#define Alignment_ReferenceTrajectories_CosmicsReferenceTrajectory_h 

/**
 *  Class implementing the reference trajectory of a single charged
 *  comics particle, i.e. a straight track with 4 parameters. Given the
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
 *  By default, the mass is assumed to be the muon-mass, but can be
 *  changed via a constructor argument.
 *
 *  Since no a-priori momentum estimate exists, a fixed value is used (that
 *  represents for example the average momentum of the cosmics spectrum).
 *
 * LIMITATIONS:
 *  So far all input hits have to be valid, but invalid hits
 *  would be needed to take into account the material effects in them...
 *
 */

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"


class CosmicsReferenceTrajectory : public ReferenceTrajectory
{

public:
  /**Constructor with Tsos at first hit (in physical order) and list of hits 
     [if (hitsAreReverse) ==> order of hits is in opposite direction compared
     to the flight of particle, but note that CosmicsReferenceTrajectory::recHits()
     returns the hits always in order of flight],
     the material effects to be considered and a particle mass and a momentum extimate,
     the magnetic field of the event is needed for propagations etc.
   */
  CosmicsReferenceTrajectory(const TrajectoryStateOnSurface &referenceTsos,
		      const TransientTrackingRecHit::ConstRecHitContainer &recHits,
		      bool hitsAreReverse,
		      const MagneticField *magField,
		      MaterialEffects materialEffects = combined, 
		      double mass = 0.10565836,
		      double momentumEstimate = 1.3 );

  virtual ~CosmicsReferenceTrajectory() {}

  virtual CosmicsReferenceTrajectory* clone() const
    { return new CosmicsReferenceTrajectory(*this); }

private:

  double theMomentumEstimate;
};

#endif
