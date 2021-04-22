#include "L1Trigger/L1THGCal/interface/HGCalModuleDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

HGCalModuleDetId::HGCalModuleDetId() : DetId() {}

HGCalModuleDetId::HGCalModuleDetId(uint32_t rawid) : DetId(rawid) {}

HGCalModuleDetId::HGCalModuleDetId(
    HGCalTriggerSubdetector subdet, int zp, int type, int layer, int sector, int moduleU, int moduleV)
    : DetId(Forward, HGCTrigger) {
  int classid = HGCalClassIdentifier::ModuleDetId;
  int zside = (zp < 0) ? 1 : 0;

  id_ |=
      (((moduleU & kHGCalModuleUMask) << kHGCalModuleUOffset) | ((moduleV & kHGCalModuleVMask) << kHGCalModuleVOffset) |
       ((sector & kHGCalSectorMask) << kHGCalSectorOffset) | ((layer & kHGCalLayerMask) << kHGCalLayerOffset) |
       ((zside & kHGCalZsideMask) << kHGCalZsideOffset) | ((type & kHGCalTypeMask) << kHGCalTypeOffset) |
       ((subdet & kHGCalTriggerSubdetMask) << kHGCalTriggerSubdetOffset) |
       ((classid & kHGCalClassIdentifierMask) << kHGCalClassIdentifierOffset));
}

HGCalModuleDetId::HGCalModuleDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCalModuleDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCalModuleDetId& HGCalModuleDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCalModuleDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const HGCalModuleDetId& id) {
  return s << "HGCalModuleDetId::HFNose:EE:HSil:HScin= " << id.isHFNose() << ":" << id.isEE() << ":" << id.isHSilicon()
           << ":" << id.isHScintillator() << " type= " << id.type() << " z= " << id.zside() << " layer= " << id.layer()
           << " sector= " << id.sector() << " module(u,v)= (" << id.moduleU() << "," << id.moduleV() << ")";
}
