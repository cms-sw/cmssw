#ifndef SeedCleanerBySharedInput_H
#define SeedCleanerBySharedInput_H
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"

/** Copy of SeedCleanerByHitPosition. Difference: uses sharedInput method
 *  of TrackingRecHit to determine equality. */

class SeedCleanerBySharedInput : public RedundantSeedCleaner  {
  public:
   /** In this implementation, it does nothing */
   virtual void add(const Trajectory *traj) { }

   /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
   virtual void init(const std::vector<Trajectory> *vect) { trajectories = vect; }

   virtual void done() { trajectories = 0; };
   
   /** \brief Returns true if the seed is not overlapping with another trajectory */
   virtual bool good(const TrajectorySeed *seed) ;

   
   SeedCleanerBySharedInput() : RedundantSeedCleaner(), trajectories(0) {}
  private:
   const std::vector<Trajectory> *trajectories; 
   
};

#endif
