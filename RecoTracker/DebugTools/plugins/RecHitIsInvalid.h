#ifndef RecHitIsInvalid_H
#define RecHitIsInvalid_H

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include <functional>

/** Predicate, returns true if RecHit of TrajectoryMeasurement is invalid
 */

class RecHitIsInvalid : public std::unary_function< const TrajectoryMeasurement&, bool> {
public:
  bool operator()( const TrajectoryMeasurement& meas) { 
    return !meas.recHit()->isValid();
  }
};

#endif
