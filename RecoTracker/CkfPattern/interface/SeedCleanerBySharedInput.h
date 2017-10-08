#ifndef SeedCleanerBySharedInput_H
#define SeedCleanerBySharedInput_H
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"

/** Copy of SeedCleanerByHitPosition. Difference: uses sharedInput method
 *  of TrackingRecHit to determine equality. */

class SeedCleanerBySharedInput : public RedundantSeedCleaner  {
  public:
   /** In this implementation, it does nothing */
   void add(const Trajectory *traj) override { }

   /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
   void init(const std::vector<Trajectory> *vect) override { trajectories = vect; }

   void done() override { trajectories = nullptr; };
   
   /** \brief Returns true if the seed is not overlapping with another trajectory */
   bool good(const TrajectorySeed *seed) override ;

   
   SeedCleanerBySharedInput() : RedundantSeedCleaner(), trajectories(nullptr) {}
  private:
   const std::vector<Trajectory> *trajectories; 
   
};

#endif
