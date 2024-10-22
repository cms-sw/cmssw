#ifndef DataFormats_TrackingRecHitSoA_SiPixelHitStatus_H
#define DataFormats_TrackingRecHitSoA_SiPixelHitStatus_H

#include <cstdint>

// more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
struct SiPixelHitStatus {
  bool isBigX : 1;   //  ∈[0,1]
  bool isOneX : 1;   //  ∈[0,1]
  bool isBigY : 1;   //  ∈[0,1]
  bool isOneY : 1;   //  ∈[0,1]
  uint8_t qBin : 3;  //  ∈[0,1,...,7]
};

struct SiPixelHitStatusAndCharge {
  SiPixelHitStatus status;
  uint32_t charge : 24;
};

#endif
