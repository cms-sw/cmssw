#ifndef FTSFromSimHitFactory_H
#define FTSFromSimHitFactory_H

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

class PSimHit;
class GeomDetUnit;
class MagneticField;

/** Produces a FreeTrajectoryState from a SimHit.
 * the FreeTrajectoryState position coinsides with the SimHit
 * position, and direction, momenta and charge are deduced
 * from the SimHit itself, without any access to the SimTrack 
 * that produced the SimHit.
 */

class FTSFromSimHitFactory {
public:

  FreeTrajectoryState operator()( const PSimHit& hit, const GeomDetUnit& det,
				  const MagneticField& field) const;

private:

  TrackCharge  charge( int particleId) const;
  
};

#endif
