#ifndef RedundantSeedCleaner_H
#define RedundantSeedCleaner_H
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include <vector>

class RedundantSeedCleaner {
  public:
  virtual ~RedundantSeedCleaner(){}
   /** \brief Informs the cleaner that a new trajectory has been made, in case the cleaner keeps a local collection of those tracks (i.e. in a map) */
   virtual void add(const Trajectory *traj) = 0;

   /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
   virtual void init(const std::vector<Trajectory> *vect) = 0;

   /** \brief Returns true if the seed is not overlapping with another trajectory */
   virtual bool good(const TrajectorySeed *seed) = 0;
   
   /** \brief Tells the cleaner that the seeds are finished, and so it can clear any cache it has */
   virtual void done() = 0;
};
#endif
