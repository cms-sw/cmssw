#ifndef CondFormats_EcalObjects_EcalRecHitParameters_h
#define CondFormats_EcalObjects_EcalRecHitParameters_h

#include <array>
#include <bitset>
#include <cstdint>

constexpr size_t kNEcalChannelStatusCodes = 16;  // The HW supports 16 channel status codes
using RecoFlagBitsArray =
    std::array<std::uint32_t, kNEcalChannelStatusCodes>;  // associate recoFlagBits to all channel status codes

struct EcalRecHitParameters {
  RecoFlagBitsArray recoFlagBits;
  std::bitset<kNEcalChannelStatusCodes> channelStatusCodesToBeExcluded;
};

#endif
