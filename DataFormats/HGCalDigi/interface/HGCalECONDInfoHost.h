#ifndef DataFormats_HGCalDigi_interface_HGCalECONDInfoHost_h
#define DataFormats_HGCalDigi_interface_HGCalECONDInfoHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDInfoSoA.h"

namespace hgcaldigi {

  // SoA with x, y, z, id fields in host memory
  using HGCalECONDInfoHost = PortableHostCollection<HGCalECONDInfoSoA>;

}  // namespace hgcaldigi

#endif
