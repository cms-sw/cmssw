#ifndef DataFormats_FTLRecHitComparison_H
#define DataFormats_FTLRecHitComparison_H

#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"

//ordering capability mandatory for lazy getter framework
// Comparison operators
inline bool operator<(const FTLRecHit& one, const FTLRecHit& other) {
  if (one.detid() == other.detid()) {
    return one.energy() < other.energy();
  }
  return one.detid() < other.detid();
}

inline bool operator<(const FTLRecHit& one, const uint32_t& detid) { return one.detid() < detid; }

inline bool operator<(const uint32_t& detid, const FTLRecHit& other) { return detid < other.detid(); }

#endif
