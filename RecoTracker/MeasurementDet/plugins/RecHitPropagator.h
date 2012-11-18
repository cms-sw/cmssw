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
TrajectoryStateOnSurface fastProp(const TrajectoryStateOnSurface& ts, const Plane& oPlane, const Plane& tPlane) {
  GlobalVector gdir = ts.globalMomentum();
   
  double delta = tPlane.localZ(oPlane.position());
  LocalVector ldir = tPlane.toLocal(gdir);  // fast prop!
  LocalPoint lPos = tPlane.toLocal( ts.globalPosition());
  LocalPoint projectedPos = lPos - ldir * delta/ldir.z();
  GlobalTrajectoryParameters gp(tPlane.toGlobal(projectedPos),gdir,ts.charge(), &ts.globalParameters().magneticField());
  return TrajectoryStateOnSurface(gp,ts.curvilinearError(),tPlane);


}

#endif 
