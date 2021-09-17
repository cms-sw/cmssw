#ifndef CondFormats_HcalObjects_interface_HcalConvertedEffectivePedestalsGPU_h
#define CondFormats_HcalObjects_interface_HcalConvertedEffectivePedestalsGPU_h

#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalsGPU.h"

// Separate access to effective and regular pedestals
// No need to transfer/rearrange effective or vice versa if they are not going
// to be used
class HcalConvertedEffectivePedestalsGPU final : public HcalConvertedPedestalsGPU {
public:
  using HcalConvertedPedestalsGPU::HcalConvertedPedestalsGPU;
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalsGPU_h
