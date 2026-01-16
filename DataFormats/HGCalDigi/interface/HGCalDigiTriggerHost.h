#ifndef DataFormats_HGCalDigi_interface_HGCalDigiTriggerHost_h
#define DataFormats_HGCalDigi_interface_HGCalDigiTriggerHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiTriggerSoA.h"

namespace hgcaldigi {

  // SoA with x, y, z, id fields in host memory
  using HGCalDigiTriggerHost = PortableHostCollection<HGCalDigiTriggerSoA>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalDigiTriggerHost_h
