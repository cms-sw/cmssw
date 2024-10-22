#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

HGCalTriggerBackendDetId::HGCalTriggerBackendDetId() : DetId() {}

HGCalTriggerBackendDetId::HGCalTriggerBackendDetId(uint32_t rawid) : DetId(rawid) {}

HGCalTriggerBackendDetId::HGCalTriggerBackendDetId(int zp, int type, int sector, int label)
    : DetId(Forward, HGCTrigger) {
  int classid = HGCalTriggerClassIdentifier::BackendDetId;
  int zside = (zp < 0) ? 1 : 0;
  id_ |= (((label & kHGCalLabelMask) << kHGCalLabelOffset) | ((sector & kHGCalSectorMask) << kHGCalSectorOffset) |
          ((zside & kHGCalZsideMask) << kHGCalZsideOffset) | ((type & kHGCalTypeMask) << kHGCalTypeOffset) |
          ((classid & kHGCalTriggerClassIdentifierMask) << kHGCalTriggerClassIdentifierOffset));
}

HGCalTriggerBackendDetId::HGCalTriggerBackendDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCalTriggerBackendDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCalTriggerBackendDetId& HGCalTriggerBackendDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCalTriggerBackendDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const HGCalTriggerBackendDetId& id) {
  return s << "HGCalTriggerBackendDetId::lpGBT:Stage1 FPGA:Stage2 FPGA= " << id.isLpGBT() << ":" << id.isStage1FPGA()
           << ":" << id.isStage1Link() << ":" << id.isStage2FPGA() << " z= " << id.zside() << " sector= " << id.sector()
           << " id= " << id.label();
}
