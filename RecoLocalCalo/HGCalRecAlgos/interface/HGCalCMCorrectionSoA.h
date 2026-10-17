#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcalcmml {

  GENERATE_SOA_LAYOUT(HGCalCMCorrectionSoALayout,
                      SOA_COLUMN(float, correction))  // additive ADC correction from DNN, per channel

  using HGCalCMCorrectionSoA = HGCalCMCorrectionSoALayout<>;

}  // namespace hgcalcmml

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMCorrectionSoA_h
