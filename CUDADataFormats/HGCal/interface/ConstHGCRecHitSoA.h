#ifndef CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
#define CUDADataFormats_HGCal_ConstHGCRecHitSoA_h

#include <cstdint>

class ConstHGCRecHitSoA {  //const version of the HGCRecHit class (data in the event should be immutable)
public:
  float const *energy_;       //calibrated energy of the rechit
  float const *time_;         //time jitter of the UncalibRecHit
  float const *timeError_;    //time resolution
  uint32_t const *id_;        //rechit detId
  uint32_t const *flagBits_;  //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)
  uint8_t const *son_;        //signal over noise
};

#endif  //CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
