#ifndef CondFormats_HGCalObjects_interface_HGCalConfigParamSoA_h
#define CondFormats_HGCalObjects_interface_HGCalConfigParamSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

//#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

namespace hgcalrechit {

  // Generate structure of ROC-level arrays (SoA) layout with RecHit dataformat
  GENERATE_SOA_LAYOUT(HGCalConfigParamSoALayout,
                      SOA_COLUMN(uint8_t, gain)  // for ADC to charge (fC) conversion (80, 160, 320 fC)
  )
  using HGCalConfigParamSoA = HGCalConfigParamSoALayout<>;

}  // namespace hgcalrechit

#endif  // CondFormats_HGCalObjects_interface_HGCalConfigParamSoA_h
