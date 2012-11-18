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

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

// propagate from glued to mono/stereo
inline
TrajectoryStateOnSurface fastProp(const TrajectoryStateOnSurface& ts, const GeomDet& origin, const GeomDet& target) {
  GlobalVector gdir = ts.globalMomentum();
  const BoundPlane& oPlane = origin.surface();
  const BoundPlane& tPlane = target.surface();
    
  double delta = tPlane.localZ(oPlane.position());
  LocalVector ldir = tPlane.toLocal(gdir);  // fast prop!
  LocalPoint lPos = tPlane.toLocal( ts.globalPosition());
  LocalPoint projectedPos = lPos - ldir * delta/ldir.z();
  GlobalTrajectoryParameters gp(tPlane.toGlobal(projectedPos),gdir,ts.charge(), &ts.globalParameters().magneticField());
  return TrajectoryStateOnSurface(gp,ts.curvilinearError(),tPlane);


}

#endif 
