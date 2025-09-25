#ifndef DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h
#define DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {
  // Generate structure of arrays (SoA) layout with Digi dataformat
  GENERATE_SOA_LAYOUT(HGCalDigiTriggerSoALayout,
                      SOA_COLUMN(uint8_t, algo),
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(uint8_t, BXm3_location),
                      SOA_COLUMN(uint8_t, BXm2_location),
                      SOA_COLUMN(uint8_t, BXm1_location),
                      SOA_COLUMN(uint8_t, BX0_location),
                      SOA_COLUMN(uint8_t, BXp1_location),
                      SOA_COLUMN(uint8_t, BXp2_location),
                      SOA_COLUMN(uint8_t, BXp3_location),
                      SOA_COLUMN(uint16_t, BXm3_energy),
                      SOA_COLUMN(uint16_t, BXm2_energy),
                      SOA_COLUMN(uint16_t, BXm1_energy),
                      SOA_COLUMN(uint16_t, BX0_energy),
                      SOA_COLUMN(uint16_t, BXp1_energy),
                      SOA_COLUMN(uint16_t, BXp2_energy),
                      SOA_COLUMN(uint16_t, BXp3_energy),
                      SOA_COLUMN(uint16_t, flags),
                      SOA_COLUMN(uint16_t, layer),
                      SOA_COLUMN(uint16_t, moduleIdx))
  using HGCalDigiTriggerSoA = HGCalDigiTriggerSoALayout<>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h
