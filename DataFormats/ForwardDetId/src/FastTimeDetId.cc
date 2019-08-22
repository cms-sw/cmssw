#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const FastTimeDetId FastTimeDetId::Undefined(0, 0, 0, 0);

FastTimeDetId::FastTimeDetId() : DetId() {}

FastTimeDetId::FastTimeDetId(uint32_t rawid) : DetId(rawid) {}

FastTimeDetId::FastTimeDetId(int module_type, int module_iz, int module_iphi, int module_zside)
    : DetId(Forward, FastTime) {
  int zsid = (module_zside > 0) ? (kFastTimeZsideMask) : (0);
  id_ |= (((module_type & kFastTimeTypeMask) << kFastTimeTypeOffset) |
          ((module_iz & kFastTimeCellZMask) << kFastTimeCellZOffset) |
          ((module_iphi & kFastTimeCellPhiMask) << kFastTimeCellPhiOffset) | (zsid << kFastTimeZsideOffset));
}

FastTimeDetId::FastTimeDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet = (ForwardSubdetector(gen.subdetId()));
    if (gen.det() != Forward || (subdet != FastTime)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize FastTimeDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

FastTimeDetId& FastTimeDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet = (ForwardSubdetector(gen.subdetId()));
    if (gen.det() != Forward || (subdet != FastTime)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign FastTimeDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const FastTimeDetId& id) {
  return s << "(FastTime " << id.type() << ", iz " << ((id.zside() > 0) ? ("+ ") : ("- ")) << " iz/ieta " << id.iz()
           << ", iphi " << id.iphi() << ")";
}
