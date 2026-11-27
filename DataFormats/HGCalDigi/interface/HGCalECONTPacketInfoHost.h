#ifndef DataFormats_HGCalDigi_interface_HGCalECONTPacketInfoHost_h
#define DataFormats_HGCalDigi_interface_HGCalECONTPacketInfoHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONTPacketInfoSoA.h"

namespace hgcaldigi {

  // SoA with ECON-T packet info fields in host memory
  using HGCalECONTPacketInfoHost = PortableHostCollection<HGCalECONTPacketInfoSoA>;

}  // namespace hgcaldigi

#endif
