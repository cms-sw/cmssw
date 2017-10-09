#ifndef CachingSeedCleanerByHitPosition_H
#define CachingSeedCleanerByHitPosition_H
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include <map>

class CachingSeedCleanerByHitPosition : public RedundantSeedCleaner  {
  public:

   /** In this implementation, it does nothing */
   virtual void add(const Trajectory *traj) ;

   /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
   virtual void init(const std::vector<Trajectory> *vect) ;

   virtual void done() ;
   
   /** \brief Returns true if the seed is not overlapping with another trajectory */
   virtual bool good(const TrajectorySeed *seed) ;

   CachingSeedCleanerByHitPosition() : RedundantSeedCleaner(), theVault(), theCache()
                                            /*, comps_(0), tracks_(0), calls_(0)*/ {}
   virtual ~CachingSeedCleanerByHitPosition() { theVault.clear(); theCache.clear(); }
  private:
    std::vector<Trajectory::RecHitContainer> theVault;
    std::multimap<uint32_t, unsigned int> theCache;

    //uint64_t comps_, tracks_, calls_;
};

#endif
