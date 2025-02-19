#include "RecHitPropagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

TrajectoryStateOnSurface 
RecHitPropagator::propagate( const TransientTrackingRecHit& hit,
			     const Plane& plane, 
			     const TrajectoryStateOnSurface& ts) const
{
  const MagneticField& field = ts.globalParameters().magneticField();
  AnalyticalPropagator prop( &field, anyDirection);
  TrajectoryStateOnSurface tsNoErr = TrajectoryStateOnSurface( ts.globalParameters(), ts.surface());
  TrajectoryStateOnSurface hitts = prop.propagate( tsNoErr, hit.det()->specificSurface());

  // LocalVector ldir = hit.det()->specificSurface().toLocal(ts.globalMomentum());
  LocalVector ldir = hitts.localMomentum();
  LocalTrajectoryParameters ltp( hit.localPosition(), ldir, ts.charge());
  AlgebraicSymMatrix55 m;
  LocalError lhe = hit.localPositionError();
  m[3][3] = lhe.xx();
  m[3][4] = lhe.xy();
  m[4][4] = lhe.yy();

  const double epsilon = 1.e-8; // very small errors on momentum and angle
  m[0][0] = epsilon;
  m[1][1] = epsilon;
  m[2][2] = epsilon;
  LocalTrajectoryError lte( m);

  TrajectoryStateOnSurface startingState( ltp, lte, hit.det()->specificSurface(), &field);

  return prop.propagate( startingState, plane);
}
