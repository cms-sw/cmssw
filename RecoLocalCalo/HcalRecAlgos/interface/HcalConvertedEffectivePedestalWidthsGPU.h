#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalWidthsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalWidthsGPU_h

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalWidthsGPU.h"

// similar to converted effective pedestals
class HcalConvertedEffectivePedestalWidthsGPU final : public HcalConvertedPedestalWidthsGPU {
public:
  using HcalConvertedPedestalWidthsGPU::HcalConvertedPedestalWidthsGPU;

#ifndef __CUDACC__
  static std::string name() { return std::string{"hcalConvertedEffectivePedestalWidthsGPU"}; }
#endif
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalConvertedEffectivePedestalWidthsGPU_h
