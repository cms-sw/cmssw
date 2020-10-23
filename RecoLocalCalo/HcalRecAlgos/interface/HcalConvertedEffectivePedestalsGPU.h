#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalsGPU_h

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalsGPU.h"

// Separate access to effective and regular pedestals
// No need to transfer/rearrange effective or vice versa if they are not going
// to be used
class HcalConvertedEffectivePedestalsGPU final : public HcalConvertedPedestalsGPU {
public:
  using HcalConvertedPedestalsGPU::HcalConvertedPedestalsGPU;
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalsGPU_h
