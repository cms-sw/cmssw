#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcalcmml {

  GENERATE_SOA_LAYOUT(HGCalCMCorrectionSoALayout,
                      SOA_COLUMN(float, correction),  // DNN CM correction in pedsub ADC: adc - ped - correction
                      SOA_COLUMN(uint8_t, valid))     // 0: no DNN for this digi, CM_ALT = nominal

  using HGCalCMCorrectionSoA = HGCalCMCorrectionSoALayout<>;

}  // namespace hgcalcmml

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h
