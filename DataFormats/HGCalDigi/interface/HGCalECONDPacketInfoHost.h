#ifndef DataFormats_HGCalDigi_interface_HGCalECONDPacketInfoHost_h
#define DataFormats_HGCalDigi_interface_HGCalECONDPacketInfoHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoSoA.h"

namespace hgcaldigi {

  // SoA with x, y, z, id fields in host memory
  using HGCalECONDPacketInfoHost = PortableHostCollection<HGCalECONDPacketInfoSoA>;

}  // namespace hgcaldigi

#endif
