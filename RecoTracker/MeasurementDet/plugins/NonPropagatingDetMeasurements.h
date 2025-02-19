#ifndef NonPropagatingDetMeasurements_H
#define NonPropagatingDetMeasurements_H

#include <vector>

class TrajectoryMeasurement;
class MeasurementDet;
class TrajectoryStateOnSurface;
class MeasurementEstimator;

class NonPropagatingDetMeasurements {
public:

  std::vector<TrajectoryMeasurement> get( const MeasurementDet& det,
					  const TrajectoryStateOnSurface& stateOnThisDet,
					  const MeasurementEstimator& est) const;
};

#endif 
