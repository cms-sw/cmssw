#ifndef RecoTracker_CkfPattern_IntermediateTrajectoryCleaner_H
#define RecoTracker_CkfPattern_IntermediateTrajectoryCleaner_H

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class IntermediateTrajectoryCleaner {
  typedef TrackerTrajectoryBuilder::TempTrajectoryContainer TempTrajectoryContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer; 
public:
  static void clean(TempTrajectoryContainer &tracks) ;
};
#endif
