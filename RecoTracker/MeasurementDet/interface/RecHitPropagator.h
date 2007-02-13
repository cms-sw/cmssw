#ifndef RecHitPropagator_H
#define RecHitPropagator_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TransientTrackingRecHit;
class MagneticField;
class Plane;

class RecHitPropagator {
public:

  TrajectoryStateOnSurface propagate( const TransientTrackingRecHit& hit,
				      const Plane& plane, 
				      const TrajectoryStateOnSurface& ts) const;

};

#endif 
