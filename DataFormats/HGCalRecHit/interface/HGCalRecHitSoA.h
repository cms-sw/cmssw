#ifndef DataFormats_HGCalRecHit_interface_HGCalRecHitSoA_h
#define DataFormats_HGCalRecHit_interface_HGCalRecHitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace hgcalrechit {

  // Generate structure of arrays (SoA) layout with RecHit dataformat
  GENERATE_SOA_LAYOUT(HGCalRecHitSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(uint32_t, detid),
                      SOA_COLUMN(double  , energy),
                      SOA_COLUMN(double  , time),
                      SOA_COLUMN(uint16_t, flags)
  )
  using HGCalRecHitSoA = HGCalRecHitSoALayout<>;
  
}  // namespace hgcalrechit

#endif  // DataFormats_HGCalRecHit_interface_HGCalRecHitSoA_h
