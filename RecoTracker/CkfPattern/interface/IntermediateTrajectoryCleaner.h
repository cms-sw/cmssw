#ifndef RecoTracker_CkfPattern_IntermediateTrajectoryCleaner_H
#define RecoTracker_CkfPattern_IntermediateTrajectoryCleaner_H

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class IntermediateTrajectoryCleaner {
  typedef BaseCkfTrajectoryBuilder::TempTrajectoryContainer TempTrajectoryContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer; 
public:
  static void clean(TempTrajectoryContainer &tracks) ;
};
#endif
