#ifndef CondFormats_EcalObjects_EcalRecHitParametersSoA_h
#define CondFormats_EcalObjects_EcalRecHitParametersSoA_h

#include <bitset>
#include <array>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

constexpr size_t kNEcalChannelStatusCodes = 16;  // The HW supports 16 channel status codes
using RecoFlagBitsArray =
    std::array<uint32_t, kNEcalChannelStatusCodes>;  // associate recoFlagBits to all channel status codes

GENERATE_SOA_LAYOUT(EcalRecHitParametersSoALayout,
                    SOA_SCALAR(RecoFlagBitsArray, recoFlagBits),
                    SOA_SCALAR(std::bitset<kNEcalChannelStatusCodes>, channelStatusCodesToBeExcluded))

using EcalRecHitParametersSoA = EcalRecHitParametersSoALayout<>;

#endif
