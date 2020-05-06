#ifndef CudaDataFormats_HGCal_HGCRecHitSoA_h
#define CudaDataFormats_HGCal_HGCRecHitSoA_h

class HGCRecHitSoA {
 public:
  float *energy_; //calibrated energy of the rechit
  float *time_; //time jitter of the UncalibRecHit
  float *timeError_; //time resolution
  uint32_t *id_; //rechit detId
  uint32_t *flagBits_; //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)
  uint8_t *son_; //signal over noise
  int nbytes_; //number of bytes of the SoA
};

#endif
