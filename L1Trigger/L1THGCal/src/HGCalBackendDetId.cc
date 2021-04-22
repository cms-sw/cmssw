#include "L1Trigger/L1THGCal/interface/HGCalBackendDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

HGCalBackendDetId::HGCalBackendDetId() : DetId() {}

HGCalBackendDetId::HGCalBackendDetId(uint32_t rawid) : DetId(rawid) {}

HGCalBackendDetId::HGCalBackendDetId(int zp, int type, int sector, int label) : DetId(Forward, HGCTrigger) {
  int classid = HGCalClassIdentifier::ModuleDetId;
  int zside = (zp < 0) ? 1 : 0;
  id_ |= (((label & kHGCalLabelMask) << kHGCalLabelOffset) | ((sector & kHGCalSectorMask) << kHGCalSectorOffset) |
          ((zside & kHGCalZsideMask) << kHGCalZsideOffset) | ((type & kHGCalTypeMask) << kHGCalTypeOffset) |
          ((classid & kHGCalClassIdentifierMask) << kHGCalClassIdentifierOffset));
}

HGCalBackendDetId::HGCalBackendDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCalBackendDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCalBackendDetId& HGCalBackendDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != Forward) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCalBackendDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s, const HGCalBackendDetId& id) {
  return s << "HGCalBackendDetId::lpGBT:Stage1 FPGA:Stage2 FPGA= " << id.isLpGBT() << ":" << id.isStage1FPGA() << ":"
           << id.isStage1Link() << ":" << id.isStage2FPGA() << " z= " << id.zside() << " sector= " << id.sector()
           << " id= " << id.label();
}
