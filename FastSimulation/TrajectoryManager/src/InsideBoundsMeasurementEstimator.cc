#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
bool InsideBoundsMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts, 
					     const Plane& plane) const
{
  return plane.bounds().inside(ts.localPosition());
}

MeasurementEstimator::Local2DVector 
InsideBoundsMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							const Plane&) const
{
  return Local2DVector(0,0);
}

std::pair<bool,double> 
InsideBoundsMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
					   const TransientTrackingRecHit& aRecHit) const 
{
  bool inside = aRecHit.det()->surface().bounds().inside(tsos.localPosition());
  return HitReturnType (inside,0);
			
}
