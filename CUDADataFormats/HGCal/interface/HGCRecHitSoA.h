#ifndef CUDADATAFORMATS_HGCRECHITSOA_H
#define CUDADATAFORMATS_HGCRECHITSOA_H 1

class HGCRecHitSoA {
 public:
  float *energy;
  float *time;
  float *timeError;
  uint32_t *id;
  uint32_t *flagBits;
  uint8_t *son;
  int nbytes;
};

#endif //CUDADATAFORMATS_HGCRECHITSOA_H
