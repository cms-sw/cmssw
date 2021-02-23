#ifndef CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
#define CUDADataFormats_HGCal_ConstHGCRecHitSoA_h

class ConstHGCRecHitSoA { //const version of the HGCRecHit class (data in the event should be immutable)
public:
  float const *energy_;            //calibrated energy of the rechit
  float const *time_;              //time jitter of the UncalibRecHit
  float const *timeError_;         //time resolution
  std::uint32_t const *id_;        //rechit detId
  std::uint32_t const *flagBits_;  //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)
  std::uint8_t const *son_;        //signal over noise

  std::uint32_t nbytes_;  //number of bytes of the SoA
  std::uint32_t nhits_;   //number of hits stored in the SoA
  std::uint32_t pad_;     //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
};

#endif //CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
