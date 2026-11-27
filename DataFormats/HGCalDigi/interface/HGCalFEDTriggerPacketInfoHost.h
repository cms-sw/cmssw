#ifndef DataFormats_HGCalDigi_interface_HGCalFEDTriggerPacketInfoHost_h
#define DataFormats_HGCalDigi_interface_HGCalFEDTriggerPacketInfoHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDTriggerPacketInfoSoA.h"

namespace hgcaldigi {

  using HGCalFEDTriggerPacketInfoHost = PortableHostCollection<HGCalFEDTriggerPacketInfoSoA>;

}  // namespace hgcaldigi

#endif
