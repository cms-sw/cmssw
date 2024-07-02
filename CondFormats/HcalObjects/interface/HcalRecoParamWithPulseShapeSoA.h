#ifndef CondFormats_HcalObjects_HcalRecoParamWithPulseShapeSoA_h
#define CondFormats_HcalObjects_HcalRecoParamWithPulseShapeSoA_h

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

#include <array>

namespace hcal {
  using HcalPSfunctorArray = std::array<float, hcal::constants::maxPSshapeBin>;  // 256
  using HcalPSfunctorBXarray = std::array<float, hcal::constants::nsPerBX>;      // 25

  GENERATE_SOA_LAYOUT(HcalRecoParamSoALayout,
                      SOA_COLUMN(uint32_t, param1),
                      SOA_COLUMN(uint32_t, param2),
                      SOA_COLUMN(uint32_t, ids))
  GENERATE_SOA_LAYOUT(HcalPulseShapeSoALayout,
                      SOA_COLUMN(HcalPSfunctorArray, acc25nsVec),
                      SOA_COLUMN(HcalPSfunctorArray, diff25nsItvlVec),
                      SOA_COLUMN(HcalPSfunctorBXarray, accVarLenIdxMinusOneVec),
                      SOA_COLUMN(HcalPSfunctorBXarray, diffVarItvlIdxMinusOneVec),
                      SOA_COLUMN(HcalPSfunctorBXarray, accVarLenIdxZEROVec),
                      SOA_COLUMN(HcalPSfunctorBXarray, diffVarItvlIdxZEROVec))

  using HcalRecoParamSoA = HcalRecoParamSoALayout<>;
  using HcalPulseShapeSoA = HcalPulseShapeSoALayout<>;
}  // namespace hcal
#endif
