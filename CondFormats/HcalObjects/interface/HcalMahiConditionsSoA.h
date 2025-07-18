#ifndef CondFormats_HcalObjects_HcalMahiConditionsSoA_h
#define CondFormats_HcalObjects_HcalMahiConditionsSoA_h

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include <array>

namespace hcal {

  static constexpr uint32_t numValuesPerChannel = 16;
  using HcalPedestalArray = std::array<float, 4>;                     // 4 capIds
  using HcalQIECodersArray = std::array<float, numValuesPerChannel>;  // QIEData

  GENERATE_SOA_LAYOUT(HcalMahiConditionsSoALayout,
                      SOA_COLUMN(HcalPedestalArray, pedestals_value),
                      SOA_COLUMN(HcalPedestalArray, pedestals_width),
                      SOA_COLUMN(HcalPedestalArray, gains_value),
                      SOA_COLUMN(HcalPedestalArray, effectivePedestals),
                      SOA_COLUMN(HcalPedestalArray, effectivePedestalWidths),
                      SOA_COLUMN(float, lutCorrs_values),
                      SOA_COLUMN(float, respCorrs_values),
                      SOA_COLUMN(float, timeCorrs_values),
                      // Use EIGEN_COLUMN for matrix?
                      SOA_COLUMN(float, pedestalWidths_sigma00),
                      SOA_COLUMN(float, pedestalWidths_sigma01),
                      SOA_COLUMN(float, pedestalWidths_sigma02),
                      SOA_COLUMN(float, pedestalWidths_sigma03),
                      SOA_COLUMN(float, pedestalWidths_sigma10),
                      SOA_COLUMN(float, pedestalWidths_sigma11),
                      SOA_COLUMN(float, pedestalWidths_sigma12),
                      SOA_COLUMN(float, pedestalWidths_sigma13),
                      SOA_COLUMN(float, pedestalWidths_sigma20),
                      SOA_COLUMN(float, pedestalWidths_sigma21),
                      SOA_COLUMN(float, pedestalWidths_sigma22),
                      SOA_COLUMN(float, pedestalWidths_sigma23),
                      SOA_COLUMN(float, pedestalWidths_sigma30),
                      SOA_COLUMN(float, pedestalWidths_sigma31),
                      SOA_COLUMN(float, pedestalWidths_sigma32),
                      SOA_COLUMN(float, pedestalWidths_sigma33),
                      SOA_COLUMN(float, gainWidths_value0),
                      SOA_COLUMN(float, gainWidths_value1),
                      SOA_COLUMN(float, gainWidths_value2),
                      SOA_COLUMN(float, gainWidths_value3),
                      SOA_COLUMN(uint32_t, channelQuality_status),
                      SOA_COLUMN(HcalQIECodersArray, qieCoders_offsets),
                      SOA_COLUMN(HcalQIECodersArray, qieCoders_slopes),
                      SOA_COLUMN(int, qieTypes_values),
                      SOA_COLUMN(int, sipmPar_type),
                      SOA_COLUMN(int, sipmPar_auxi1),
                      SOA_COLUMN(float, sipmPar_fcByPE),
                      SOA_COLUMN(float, sipmPar_darkCurrent),
                      SOA_COLUMN(float, sipmPar_auxi2),
                      SOA_SCALAR(int, maxDepthHB),
                      SOA_SCALAR(int, maxDepthHE),
                      SOA_SCALAR(int, maxPhiHE),
                      SOA_SCALAR(int, firstHBRing),
                      SOA_SCALAR(int, lastHBRing),
                      SOA_SCALAR(int, firstHERing),
                      SOA_SCALAR(int, lastHERing),
                      SOA_SCALAR(int, nEtaHB),
                      SOA_SCALAR(int, nEtaHE),
                      SOA_SCALAR(uint32_t, offsetForHashes))
  using HcalMahiConditionsSoA = HcalMahiConditionsSoALayout<>;

}  // namespace hcal

#endif
