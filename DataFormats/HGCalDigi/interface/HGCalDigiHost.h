#ifndef DataFormats_HGCalDigi_interface_HGCalDigiHost_h
#define DataFormats_HGCalDigi_interface_HGCalDigiHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiSoA.h"

namespace hgcaldigi {

  // SoA with x, y, z, id fields in host memory
  using HGCalDigiHost = PortableHostCollection<HGCalDigiSoA>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalDigiHost_h
