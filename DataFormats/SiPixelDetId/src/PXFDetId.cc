#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

PXFDetId::PXFDetId() : DetId() {}

PXFDetId::PXFDetId(uint32_t rawid) : DetId(rawid) {}
PXFDetId::PXFDetId(const DetId& id) : DetId(id.rawId()) {}

std::ostream& operator<<(std::ostream& os, const PXFDetId& id) {
  return os << "(PixelEndcap " << id.disk() << ',' << id.blade() << ',' << id.panel() << ',' << id.module() << ')';
}
