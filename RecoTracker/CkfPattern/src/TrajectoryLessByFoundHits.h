#ifndef TrajectoryLessByFoundHits_h_
#define TrajectoryLessByFoundHits_h_

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <TrackingTools/PatternTools/interface/TempTrajectory.h>

struct TrajectoryLessByFoundHits {
  bool operator() (const Trajectory& a, const Trajectory& b) const {
    return a.foundHits()<b.foundHits();
  }
  bool operator() (const TempTrajectory& a, const TempTrajectory& b) const {
    return a.foundHits()<b.foundHits();
  }
};
#endif

