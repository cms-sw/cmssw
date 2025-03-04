#ifndef CondFormats_HGCalObjects_interface_HGCalCalibParamSoA_h
#define CondFormats_HGCalObjects_interface_HGCalCalibParamSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

//#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

namespace hgcalrechit {
  using Vector32f = Eigen::Matrix<float, 32, 1>;
  using Vector8f = Eigen::Matrix<float, 8, 1>;
  using Vector3f = Eigen::Matrix<float, 3, 1>;
  constexpr float TOAtops = 24.41f;

  // Generate structure of channel-level arrays (SoA) layout with RecHit dataformat
  GENERATE_SOA_LAYOUT(HGCalCalibParamSoALayout,
                      SOA_COLUMN(float, ADC_ped),             // ADC pedestals, O(91)
                      SOA_COLUMN(float, Noise),               // noise, O(3)
                      SOA_COLUMN(float, CM_slope),            // common mode slope, O(0.25)
                      SOA_COLUMN(float, CM_ped),              // common mode pedestal (offset), O(92)
                      SOA_COLUMN(float, BXm1_slope),          // leakage correction from previous bunch, O(0.0)
                      SOA_COLUMN(float, TOTtoADC),            // TOT linearization in ADC units, O(15)
                      SOA_COLUMN(float, TOT_ped),             // TOT pedestal (offset), O(9.0)
                      SOA_COLUMN(float, TOT_lin),             // threshold at which TOT is linear, O(200)
                      SOA_COLUMN(float, TOT_P0),              // coefficient pol2 in nonlinear region, O(145)
                      SOA_COLUMN(float, TOT_P1),              // coefficient pol2 in nonlinear region, O(1.0)
                      SOA_COLUMN(float, TOT_P2),              // coefficient pol2 in nonlinear region, O(0.004)
                      SOA_EIGEN_COLUMN(Vector32f, TOA_CTDC),  // TOA coarse TDC correction
                      SOA_EIGEN_COLUMN(Vector8f, TOA_FTDC),   // TOA fine TDC correction
                      SOA_EIGEN_COLUMN(Vector3f, TOA_TW),     // TOA timewalk correction
                      SOA_COLUMN(float, MIPS_scale),          // MIPS scale
                      SOA_COLUMN(unsigned char, valid)        // only 1 bit used: if false = mask dead channel
  )
  using HGCalCalibParamSoA = HGCalCalibParamSoALayout<>;

}  // namespace hgcalrechit

#endif  // CondFormats_HGCalObjects_interface_HGCalCalibParamSoA_h
