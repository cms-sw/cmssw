#ifndef InsideBoundsMeasurementEstimator_H
#define InsideBoundsMeasurementEstimator_H

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class InsideBoundsMeasurementEstimator : public MeasurementEstimator {
public:

  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			 const Plane& plane) const;

  std::pair<bool,double> 
    estimate(const TrajectoryStateOnSurface& tsos,
	     const TrackingRecHit& aRecHit) const; 

  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const Plane& plane) const;

  virtual MeasurementEstimator* clone() const {
    return new InsideBoundsMeasurementEstimator( *this);
  }

};

#endif
