#ifndef CachingSeedCleanerBySharedInput_H
#define CachingSeedCleanerBySharedInput_H
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include <map>
#include <unordered_map>

/** Merge of SeedCleanerBySharedInput and CachingSeedCleanerByHitPosition */
class CachingSeedCleanerBySharedInput final : public RedundantSeedCleaner {
public:
  // in this implementation it populate the cache
  void add(const Trajectory *traj) override;

  /** \brief Provides the cleaner a pointer to the vector where trajectories are stored, in case it does not want to keep a local collection of trajectories */
  void init(const std::vector<Trajectory> *vect) override;

  void done() override;

  /** \brief Returns true if the seed is not overlapping with another trajectory */
  bool good(const TrajectorySeed *seed) override;

  CachingSeedCleanerBySharedInput(unsigned int numHitsForSeedCleaner = 4, bool onlyPixelHits = false)
      : theNumHitsForSeedCleaner(numHitsForSeedCleaner), theOnlyPixelHits(onlyPixelHits) {}

private:
  std::vector<Trajectory::RecHitContainer> theVault;
  std::unordered_multimap<unsigned int, unsigned int> theCache;

  int theNumHitsForSeedCleaner;
  bool theOnlyPixelHits;

  //uint64_t comps_, tracks_, calls_;
};

#endif
