#ifndef CUDADataFormats_HGCal_HGCRecHitSoA_h
#define CUDADataFormats_HGCal_HGCRecHitSoA_h

class HGCRecHitSoA {
public:
  float *energy_;       //calibrated energy of the rechit
  float *time_;         //time jitter of the UncalibRecHit
  float *timeError_;    //time resolution
  uint32_t *id_;        //rechit detId
  uint32_t *flagBits_;  //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)
  uint8_t *son_;        //signal over noise

  uint32_t nbytes_;  //number of bytes of the SoA
  uint32_t nhits_;   //number of hits stored in the SoA
  uint32_t stride_;  //stride of memory block (used for warp alignment, slighlty larger than 'nhits_')
};

namespace memory {
  namespace npointers {
    constexpr unsigned int float_hgcrechits_soa = 3;   //number of float pointers in the rechits SoA
    constexpr unsigned int uint32_hgcrechits_soa = 2;  //number of uint32_t pointers in the rechits SoA
    constexpr unsigned int uint8_hgcrechits_soa = 1;   //number of uint8_t pointers in the rechits SoA
    constexpr unsigned int ntypes_hgcrechits_soa = 3;  //number of different pointer types in the rechits SoA
  }                                                    // namespace npointers
}  // namespace memory

#endif
