#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const uint32_t HGCTriggerDetId::cell_shift;
const uint32_t HGCTriggerDetId::cell_mask;
const uint32_t HGCTriggerDetId::module_mask;
const uint32_t HGCTriggerDetId::module_shift;
const uint32_t HGCTriggerDetId::sector_shift;
const uint32_t HGCTriggerDetId::sector_mask;
const uint32_t HGCTriggerDetId::layer_shift;
const uint32_t HGCTriggerDetId::layer_mask;
const uint32_t HGCTriggerDetId::zside_shift;
const uint32_t HGCTriggerDetId::zside_mask;

const HGCTriggerDetId HGCTriggerDetId::Undefined(ForwardEmpty, 0, 0, 0, 0, 0);

HGCTriggerDetId::HGCTriggerDetId() : DetId() {}

HGCTriggerDetId::HGCTriggerDetId(uint32_t rawid) : DetId(rawid) {}

HGCTriggerDetId::HGCTriggerDetId(ForwardSubdetector subdet, int zp, int lay, int sec, int mod, int cell)
    : DetId(Forward, subdet) {
  if (zp < 0)
    zp = 0;

  setMaskedId(cell, cell_shift, cell_mask);
  setMaskedId(sec, sector_shift, sector_mask);
  setMaskedId(mod, module_shift, module_mask);
  setMaskedId(lay, layer_shift, layer_mask);
  setMaskedId(zp, zside_shift, zside_mask);
}

HGCTriggerDetId::HGCTriggerDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet = (ForwardSubdetector(gen.subdetId()));
    if (gen.det() != Forward || (subdet != HGCTrigger)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCTriggerDetId& HGCTriggerDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet = (ForwardSubdetector(gen.subdetId()));
    if (gen.det() != Forward || (subdet != HGCTrigger)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const HGCTriggerDetId& id) {
  switch (id.subdet()) {
    case (HGCTrigger):
      return s << "isEE=" << id.isEE() << " zpos=" << id.zside() << " layer=" << id.layer() << " module=" << id.module()
               << " sector=" << id.sector() << " cell=" << id.cell();
    default:
      return s << id.rawId();
  }
}
