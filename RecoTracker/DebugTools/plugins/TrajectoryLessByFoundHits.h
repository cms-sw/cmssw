#ifndef TrajectoryLessByFoundHits_h_
#define TrajectoryLessByFoundHits_h_

#include "TrackingTools/PatternTools/interface/Trajectory.h"

struct TrajectoryLessByFoundHits {
  bool operator() (const Trajectory& a, const Trajectory& b) const {
    return a.foundHits()<b.foundHits();
  }
};
#endif

