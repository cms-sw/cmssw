#ifndef HGCRecHitComparison_H
#define HGCRecHitComparison_H

//ordering capability mandatory for lazy getter framework
// Comparison operators
inline bool operator<( const HGCRecHit& one, const HGCRecHit& other) {
  if(one.detid() == other.detid()){ return one.energy() < other.energy(); }
  return one.detid() < other.detid();}

inline bool operator<( const HGCRecHit& one, const uint32_t& detid) {
  return one.detid() < detid;}

inline bool operator<( const uint32_t& detid, const HGCRecHit& other) { 
  return detid < other.detid();}

#endif

