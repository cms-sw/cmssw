#ifndef TSOSFromSimHitFactory_H
#define TSOSFromSimHitFactory_H

#include "RecoTracker/DebugTools/interface/FTSFromSimHitFactory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

class SimHit;
class MagneticField;

/** Produces a TrajectoryStateOnSurface from a SimHit.
 * the TrajectoryStateOnSurface position coinsides with the SimHit
 * position, and direction, momenta and charge are deduced
 * from the SimHit itself, without any access to the SimTrack 
 * that produced the SimHit. The surface of the result is 
 * the surface of the Det of the SimHit.
 */

class TSOSFromSimHitFactory {
public:

  TrajectoryStateOnSurface operator()( const PSimHit& hit, const GeomDetUnit& det,
				       const MagneticField& field) const {
    return TrajectoryStateOnSurface( FTSFromSimHitFactory()( hit, det, field),
				     det.surface());
  }
};

#endif
