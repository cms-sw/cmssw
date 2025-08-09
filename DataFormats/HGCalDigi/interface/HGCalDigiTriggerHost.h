#ifndef HGCalCommissioning_HGCalDigiTrigger_interface_HGCalDigiTriggerHost_h
#define HGCalCommissioning_HGCalDigiTrigger_interface_HGCalDigiTriggerHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HGCalCommissioning/HGCalDigiTrigger/interface/HGCalDigiTriggerSoA.h"

namespace hgcaldigi {

  // SoA with x, y, z, id fields in host memory
  using HGCalDigiTriggerHost = PortableHostCollection<HGCalDigiTriggerSoA>;

}  // namespace hgcaldigi

#endif  // HGCalCommissioning_HGCalDigiTrigger_interface_HGCalDigiTriggerHost_h
