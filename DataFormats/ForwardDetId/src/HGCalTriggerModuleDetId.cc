#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

HGCalTriggerModuleDetId::HGCalTriggerModuleDetId() : DetId() {}

HGCalTriggerModuleDetId::HGCalTriggerModuleDetId(uint32_t rawid) : DetId(rawid) {}

HGCalTriggerModuleDetId::HGCalTriggerModuleDetId(
    HGCalTriggerSubdetector subdet, int zp, int type, int layer, int sector, int moduleU, int moduleV)
    : DetId(Forward, HGCTrigger) {
  int classid = HGCalTriggerClassIdentifier::ModuleDetId;
  int zside = (zp < 0) ? 1 : 0;

  id_ |=
      (((moduleU & kHGCalModuleUMask) << kHGCalModuleUOffset) | ((moduleV & kHGCalModuleVMask) << kHGCalModuleVOffset) |
       ((sector & kHGCalSectorMask) << kHGCalSectorOffset) | ((layer & kHGCalLayerMask) << kHGCalLayerOffset) |
       ((zside & kHGCalZsideMask) << kHGCalZsideOffset) | ((type & kHGCalTypeMask) << kHGCalTypeOffset) |
       ((subdet & kHGCalTriggerSubdetMask) << kHGCalTriggerSubdetOffset) |
       ((classid & kHGCalTriggerClassIdentifierMask) << kHGCalTriggerClassIdentifierOffset));
}

HGCalTriggerModuleDetId::HGCalTriggerModuleDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCalTriggerModuleDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCalTriggerModuleDetId& HGCalTriggerModuleDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCalTriggerModuleDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const HGCalTriggerModuleDetId& id) {
  return s << "HGCalTriggerModuleDetId::HFNose:EE:HSil:HScin= " << id.isHFNose() << ":" << id.isEE() << ":"
           << id.isHSilicon() << ":" << id.isHScintillator() << " type= " << id.type() << " z= " << id.zside()
           << " layer= " << id.layer() << " sector= " << id.sector() << " module(u,v)= (" << id.moduleU() << ","
           << id.moduleV() << ")";
}
