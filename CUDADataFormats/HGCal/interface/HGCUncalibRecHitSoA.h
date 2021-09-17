#ifndef CUDADataFormats_HGCal_HGCUncalibRecHitSoA_h
#define CUDADataFormats_HGCal_HGCUncalibRecHitSoA_h

#include <cstdint>

class HGCUncalibRecHitSoA {
public:
  float *amplitude_;     //uncalib rechit amplitude, i.e., the average number of MIPs
  float *pedestal_;      //reconstructed pedestal
  float *jitter_;        //reconstructed time jitter
  float *chi2_;          //chi2 of the pulse
  float *OOTamplitude_;  //out-of-time reconstructed amplitude
  float *OOTchi2_;       //out-of-time chi2
  std::uint32_t *
      flags_;  //uncalibrechit flags describing its status (DataFormats/HGCRecHit/interface/HGCUncalibRecHit.h); to be propagated to the rechits
  std::uint32_t *aux_;  //aux word; first 8 bits contain time (jitter) error
  std::uint32_t *id_;   //uncalibrechit detector id

  std::uint32_t nbytes_;  //number of bytes of the SoA
  std::uint32_t nhits_;   //number of hits stored in the SoA
  std::uint32_t pad_;     //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
};

namespace memory {
  namespace npointers {
    constexpr unsigned float_hgcuncalibrechits_soa = 6;   //number of float pointers in the uncalibrated rechits SoA
    constexpr unsigned uint32_hgcuncalibrechits_soa = 3;  //number of uint32_t pointers in the uncalibrated rechits SoA
    constexpr unsigned ntypes_hgcuncalibrechits_soa =
        2;  //number of different pointer types in the uncalibrated rechits SoA
  }         // namespace npointers
}  // namespace memory

#endif
