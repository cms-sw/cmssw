#ifndef TrajectoryCleaning_TrajectoryCleanerForShortTracks_h
#define TrajectoryCleaning_TrajectoryCleanerForShortTracks_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

/** A concrete TrajectoryCleaner that considers two trajectories
 *  to be mutually exclusive if they share more than some fraction
 *  of their hits.
 *  The "best" trajectory of each set of mutually exclusive ones
 *  is kept, the others are eliminated.
 *  The goodness of a track is defined in terms of Chi^2, number of
 *  reconstructed hits, and number of lost hits.
 */


class TrajectoryCleanerForShortTracks : public TrajectoryCleaner {

 public:

  typedef std::vector<Trajectory*> 	TrajectoryPointerContainer;

  TrajectoryCleanerForShortTracks(){};
  virtual ~TrajectoryCleanerForShortTracks(){};

  using TrajectoryCleaner::clean;
  virtual void clean( TrajectoryPointerContainer&) const;

};

#endif
