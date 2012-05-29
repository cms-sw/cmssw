#ifndef InsideBoundsMeasurementEstimator_H
#define InsideBoundsMeasurementEstimator_H

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class InsideBoundsMeasurementEstimator : public MeasurementEstimator {
public:

  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			 const BoundPlane& plane) const;

  std::pair<bool,double> 
    estimate(const TrajectoryStateOnSurface& tsos,
	     const TransientTrackingRecHit& aRecHit) const; 

  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const BoundPlane& plane) const;

  virtual MeasurementEstimator* clone() const {
    return new InsideBoundsMeasurementEstimator( *this);
  }

};

#endif
