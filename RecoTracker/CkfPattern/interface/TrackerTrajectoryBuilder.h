#ifndef RecoTracker_CkfPattern_TrackerTrajectoryBuilder_h
#define RecoTracker_CkfPattern_TrackerTrajectoryBuilder_h

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"

/** The component of track reconstruction that, strating from a seed,
 *  reconstructs all possible trajectories.
 *  The resulting trajectories may be mutually exclusive and require
 *  cleaning by a TrajectoryCleaner.
 *  The Trajectories are normally not smoothed.
 */

class TrackerTrajectoryBuilder {
public:

  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;

  virtual ~TrackerTrajectoryBuilder() {};

  virtual TrajectoryContainer trajectories(const TrajectorySeed&) const = 0;

  virtual void setEvent(const edm::Event& event) const = 0;
};


#endif
