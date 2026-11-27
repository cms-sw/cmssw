#ifndef DataFormats_HGCalDigi_interface_HGCalFEDTriggerPacketInfoSoA_h
#define DataFormats_HGCalDigi_interface_HGCalFEDTriggerPacketInfoSoA_h

#include <cstdint>  // for uint8_t

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {


  GENERATE_SOA_LAYOUT(HGCalFEDTriggerPacketInfoSoALayout,
                      //FED unpacking flag
                      SOA_COLUMN(uint16_t, FEDBX_trig))

  using HGCalFEDTriggerPacketInfoSoA = HGCalFEDTriggerPacketInfoSoALayout<>;
}  // namespace hgcaldigi

#endif

