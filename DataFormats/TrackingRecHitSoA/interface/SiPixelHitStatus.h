#ifndef DataFormats_TrackingRecHitSoA_interface_SiPixelHitStatus_h
#define DataFormats_TrackingRecHitSoA_interface_SiPixelHitStatus_h

#include <cstdint>

// more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
struct SiPixelHitStatus {
  bool isBigX : 1;   //  ∈[0,1]
  bool isOneX : 1;   //  ∈[0,1]
  bool isBigY : 1;   //  ∈[0,1]
  bool isOneY : 1;   //  ∈[0,1]
  uint8_t qBin : 3;  //  ∈[0,1,...,7]
};

static_assert(sizeof(SiPixelHitStatus) == sizeof(uint8_t));
static_assert(alignof(SiPixelHitStatus) == alignof(uint8_t));

struct alignas(alignof(uint32_t)) SiPixelHitStatusAndCharge {
#ifdef __CLING__
  // ROOT does not support the serialisation and deserialisation of bit fields.
  // See https://github.com/root-project/root/issues/17501 .
  uint8_t status;
  uint8_t charge[3];
#else
  SiPixelHitStatus status;
  uint32_t charge : 24;
#endif
};

static_assert(sizeof(SiPixelHitStatusAndCharge) == sizeof(uint32_t));
static_assert(alignof(SiPixelHitStatusAndCharge) == alignof(uint32_t));

#endif  // DataFormats_TrackingRecHitSoA_interface_SiPixelHitStatus_h
