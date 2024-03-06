#ifndef DataFormats_HGCalDigi_interface_HGCalDigiSoA_h
#define DataFormats_HGCalDigi_interface_HGCalDigiSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace hgcaldigi {

  // Generate structure of arrays (SoA) layout with Digi dataformat
  GENERATE_SOA_LAYOUT(HGCalDigiSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(uint32_t, electronicsId),
                      SOA_COLUMN(uint8_t, tctp),
                      SOA_COLUMN(uint16_t, adcm1),
                      SOA_COLUMN(uint16_t, adc),
                      SOA_COLUMN(uint16_t, tot),
                      SOA_COLUMN(uint16_t, toa),
                      SOA_COLUMN(uint16_t, cm),
                      SOA_COLUMN(uint16_t, flags)
  )
  using HGCalDigiSoA = HGCalDigiSoALayout<>;
  
}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalDigiSoA_h
