#ifndef CUDADataFormats_TrackingRecHit_interface_SiPixelHitStatus_H
#define CUDADataFormats_TrackingRecHit_interface_SiPixelHitStatus_H

#include <cstdint>

// more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
struct SiPixelHitStatus {
  bool isBigX : 1;   //  ∈[0,1]
  bool isOneX : 1;   //  ∈[0,1]
  bool isBigY : 1;   //  ∈[0,1]
  bool isOneY : 1;   //  ∈[0,1]
  uint8_t qBin : 3;  //  ∈[0,1,...,7]
};

#endif
