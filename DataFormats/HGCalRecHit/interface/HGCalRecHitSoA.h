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
                      SOA_COLUMN(double, energy),
                      SOA_COLUMN(double, time),
                      SOA_COLUMN(uint16_t, flags))
  using HGCalRecHitSoA = HGCalRecHitSoALayout<>;

  enum HGCalRecHitFlags { Normal = 0x0, EnergyInvalid = 0x1, TimeInvalid = 0x2 };

}  // namespace hgcalrechit

#endif  // DataFormats_HGCalRecHit_interface_HGCalRecHitSoA_h
