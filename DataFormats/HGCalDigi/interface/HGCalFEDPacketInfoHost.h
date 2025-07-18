#ifndef DataFormats_HGCalDigi_interface_HGCalFEDPacketInfoHost_h
#define DataFormats_HGCalDigi_interface_HGCalFEDPacketInfoHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoSoA.h"

namespace hgcaldigi {

  using HGCalFEDPacketInfoHost = PortableHostCollection<HGCalFEDPacketInfoSoA>;

}  // namespace hgcaldigi

#endif
