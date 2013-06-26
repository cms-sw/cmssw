#ifndef OrderedHitSeeds_H
#define OrderedHitSeeds_H

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitSeeds : public std::vector<SeedingHitSet>, public OrderedSeedingHits {
public:

  virtual ~OrderedHitSeeds(){}

  virtual unsigned int size() const { return std::vector<SeedingHitSet>::size(); }

  virtual const SeedingHitSet & operator[](unsigned int i) const {
    return std::vector<SeedingHitSet>::operator[](i);
  }

};
#endif
