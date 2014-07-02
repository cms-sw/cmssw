#ifndef RecHitPropagator_H
#define RecHitPropagator_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TrackingRecHit;
class MagneticField;
class Plane;

class RecHitPropagator {
public:

  TrajectoryStateOnSurface propagate( const TrackingRecHit& hit,
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
  // we can also patch it up as only the position-errors are used...
  GlobalTrajectoryParameters gp(tPlane.toGlobal(projectedPos),gdir,ts.charge(), &ts.globalParameters().magneticField());
  if (ts.hasError())
    return TrajectoryStateOnSurface(gp,ts.curvilinearError(),tPlane);
  else
    return TrajectoryStateOnSurface(gp,tPlane);


}

#endif 
