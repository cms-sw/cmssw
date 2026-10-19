#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMML_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMML_h

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcalcmml {

  // Inputs of the multi-module CM DNN for one digi, in training order (all_channels_multimodule_v1)
  namespace Feature {
    constexpr uint32_t CM0 = 0,  // [0-11]    cm_erx00..11_pedsub: 2*(0.5*cm - CM_ped), 0 if eRx off/invalid
        NCM = 12,
                       NCHTOA = 12,  // [12]      nchtoa: module channels with toa > 0
        NCHTOT = 13,                 // [13]      nchtot: module channels in TOT mode with tot > 0
        NCHADCGT0 = 14,              // [14-17]   nchadcgt10/50/200/500: module channels with pedsub ADC > X
        NNCHADCGT = 4,
                       ONEHOT0 = 18,  // [18-28]   module_onehot_000..010 (last = UNKNOWN, never set)
        NONEHOT = 11,
                       ADC0 = 29,  // [29-250]  adc_allch_000..221: pedsub ADC, target channel's own slot 0
        NCH = 222,
                       NMODULE = 251,  // features 0..250 are identical for every digi of a module
        CHIDX = 251,                   // [251]     channel_indices: chIdx - (nch-1)/2
        ERXIDX = 252,                  // [252]     erx_indices: erxIdx - (nErx-1)/2
        CELLFRAC = 253,                // [253]     cell_area_fraction (cellareas.json SF)
        Size = 254;
  }  // namespace Feature

  constexpr float kNchAdcGtThresholds[Feature::NNCHADCGT] = {10.f, 50.f, 200.f, 500.f};

  using CMFeatureVector = Eigen::Matrix<float, Feature::Size, 1>;

  // one Eigen column: TensorCollection wraps it as a [N, 254] tensor without copying
  GENERATE_SOA_LAYOUT(HGCalCMMLSoALayout, SOA_EIGEN_COLUMN(CMFeatureVector, features))

  using HGCalCMMLSoA = HGCalCMMLSoALayout<>;

}  // namespace hgcalcmml

#endif
