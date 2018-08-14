#ifndef InsideBoundsMeasurementEstimator_H
#define InsideBoundsMeasurementEstimator_H

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class InsideBoundsMeasurementEstimator : public MeasurementEstimator {
public:

  bool estimate( const TrajectoryStateOnSurface& ts, 
			 const Plane& plane) const override;

  std::pair<bool,double> 
    estimate(const TrajectoryStateOnSurface& tsos,
	     const TrackingRecHit& aRecHit) const override; 

  Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const Plane& plane) const override;

  MeasurementEstimator* clone() const override {
    return new InsideBoundsMeasurementEstimator( *this);
  }

};

#endif
