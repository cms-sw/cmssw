#ifndef EcalRecHitComparison_H
#define EcalRecHitComparison_H

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

//ordering capability mandatory for lazy getter framework
// Comparison operators
inline bool operator<(const EcalRecHit& one, const EcalRecHit& other) {
  if (one.detid() == other.detid()) {
    return one.energy() < other.energy();
  }
  return one.detid() < other.detid();
}

inline bool operator<(const EcalRecHit& one, const uint32_t& detid) { return one.detid() < detid; }

inline bool operator<(const uint32_t& detid, const EcalRecHit& other) { return detid < other.detid(); }

#endif
