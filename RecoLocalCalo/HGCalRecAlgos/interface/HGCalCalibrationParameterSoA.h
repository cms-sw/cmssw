#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterSoA_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

namespace hgcalrechit {

  // human-readable data type to replace bool for memcpy of vector<>::.data()
  // NOTE: bool and byte have the same memory size
  using mybool = std::byte;

  // Generate structure of channel-level arrays (SoA) layout with RecHit dataformat
  GENERATE_SOA_LAYOUT(HGCalCalibParamSoALayout,
                      SOA_SCALAR(HGCalMappingModuleIndexer, map), // map elecId <-> dense idx
                      SOA_COLUMN(float, ADC_ped),     // ADC pedestals, O(91)
                      SOA_COLUMN(float, CM_slope),    // common mode slope, O(0.25)
                      SOA_COLUMN(float, CM_ped),      // common mode pedestal (offset), O(92)
                      SOA_COLUMN(float, BXm1_slope),  // leakage correction from previous bunch, O(0.0)
                      //SOA_COLUMN(float, BXm1_ped),  // redundant
                      SOA_COLUMN(float, ADCtofC),     // ADC conversion to charge (fC), depends on gain (80, 160, 320 fC)
                      SOA_COLUMN(float, TOTtofC),     // TOT conversion to charge (fC), depends on gain (80, 160, 320 fC)
                      SOA_COLUMN(float, TOT_ped),     // TOT pedestal (offset), O(9.0)
                      SOA_COLUMN(float, TOT_lin),     // threshold at which TOT is linear, O(200)
                      SOA_COLUMN(float, TOT_P0),      // coefficient pol2 in nonlinear region, O(145)
                      SOA_COLUMN(float, TOT_P1),      // coefficient pol2 in nonlinear region, O(1.0)
                      SOA_COLUMN(float, TOT_P2),      // coefficient pol2 in nonlinear region, O(0.004)
                      SOA_COLUMN(float, TOAtops),     // TOA conversion to time (ps)
                      SOA_COLUMN(mybool,valid)        // if false: mask dead channel
  )
  using HGCalCalibParamSoA = HGCalCalibParamSoALayout<>;

  // Generate structure of ROC-level arrays (SoA) layout with RecHit dataformat
  GENERATE_SOA_LAYOUT(HGCalConfigParamSoALayout,
                      SOA_SCALAR(HGCalMappingModuleIndexer, map),
                      SOA_COLUMN(uint8_t, gain)
  )
  using HGCalConfigParamSoA = HGCalConfigParamSoALayout<>;

}  // namespace hgcalrechit

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterSoA_h
