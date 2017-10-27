#ifndef SeedCleanerByHitPosition_H
#define SeedCleanerByHitPosition_H
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"

class SeedCleanerByHitPosition : public RedundantSeedCleaner  {
  public:
   /** In this implementation, it does nothing */
   void add(const Trajectory *traj) override { }

   /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
   void init(const std::vector<Trajectory> *vect) override { trajectories = vect; }

   void done() override ;
   
   /** \brief Returns true if the seed is not overlapping with another trajectory */
   bool good(const TrajectorySeed *seed) override ;

   
   SeedCleanerByHitPosition() : RedundantSeedCleaner(), trajectories(nullptr) /*,comps_(0), tracks_(0), calls_(0)*/ {}
  private:
   const std::vector<Trajectory> *trajectories; 
   //uint64_t comps_, tracks_, calls_;
   
};

#endif
