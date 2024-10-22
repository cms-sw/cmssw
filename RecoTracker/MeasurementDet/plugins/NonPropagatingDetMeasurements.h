#ifndef NonPropagatingDetMeasurements_H
#define NonPropagatingDetMeasurements_H

#include <vector>
#include "FWCore/Utilities/interface/Visibility.h"

class TrajectoryMeasurement;
class MeasurementDet;
class MeasurementTrackerEvent;
class TrajectoryStateOnSurface;
class MeasurementEstimator;

class dso_hidden NonPropagatingDetMeasurements {
public:
  std::vector<TrajectoryMeasurement> get(const MeasurementDet& det,
                                         const TrajectoryStateOnSurface& stateOnThisDet,
                                         const MeasurementEstimator& est,
                                         const MeasurementTrackerEvent& data) const;
};

#endif
