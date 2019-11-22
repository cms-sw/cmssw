#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCalTriggerDetId HGCalTriggerDetId::Undefined(HGCalEETrigger, 0, 0, 0, 0, 0, 0, 0);

HGCalTriggerDetId::HGCalTriggerDetId() : DetId() {}

HGCalTriggerDetId::HGCalTriggerDetId(uint32_t rawid) : DetId(rawid) {}

HGCalTriggerDetId::HGCalTriggerDetId(
    int subdet, int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV)
    : DetId(HGCalTrigger, ForwardEmpty) {
  int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int waferUsign = (waferU >= 0) ? 0 : 1;
  int waferVsign = (waferV >= 0) ? 0 : 1;
  int zside = (zp < 0) ? 1 : 0;
  id_ |= (((cellU & kHGCalCellUMask) << kHGCalCellUOffset) | ((cellV & kHGCalCellVMask) << kHGCalCellVOffset) |
          ((waferUabs & kHGCalWaferUMask) << kHGCalWaferUOffset) |
          ((waferUsign & kHGCalWaferUSignMask) << kHGCalWaferUSignOffset) |
          ((waferVabs & kHGCalWaferVMask) << kHGCalWaferVOffset) |
          ((waferVsign & kHGCalWaferVSignMask) << kHGCalWaferVSignOffset) |
          ((layer & kHGCalLayerMask) << kHGCalLayerOffset) | ((zside & kHGCalZsideMask) << kHGCalZsideOffset) |
          ((type & kHGCalTypeMask) << kHGCalTypeOffset) | ((subdet & kHGCalSubdetMask) << kHGCalSubdetOffset));
}

HGCalTriggerDetId::HGCalTriggerDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != HGCalTrigger) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCalTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCalTriggerDetId& HGCalTriggerDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != HGCalTrigger) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCalTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

int HGCalTriggerDetId::triggerCellX() const {
  int nT =
      (type() == HGCSiliconDetId::HGCalFine) ? HGCSiliconDetId::HGCalFineTrigger : HGCSiliconDetId::HGCalCoarseTrigger;
  int N = nT * HGCalTriggerCell;
  std::vector<int> vc = cellV();
  int x(0);
  for (auto const& v : vc) {
    x += (3 * (v - N) + 2);
  }
  return (x / static_cast<int>(vc.size()));
}

int HGCalTriggerDetId::triggerCellY() const {
  int nT =
      (type() == HGCSiliconDetId::HGCalFine) ? HGCSiliconDetId::HGCalFineTrigger : HGCSiliconDetId::HGCalCoarseTrigger;
  int N = nT * HGCalTriggerCell;
  std::vector<int> uc = cellU();
  std::vector<int> vc = cellV();
  int y(0);
  for (unsigned int k = 0; k < uc.size(); ++k) {
    y += (2 * uc[k] - (N + vc[k]));
  }
  return (y / static_cast<int>(vc.size()));
}

std::vector<int> HGCalTriggerDetId::cellU() const {
  std::vector<int> uc;
  int nT =
      (type() == HGCSiliconDetId::HGCalFine) ? HGCSiliconDetId::HGCalFineTrigger : HGCSiliconDetId::HGCalCoarseTrigger;
  if ((triggerCellU() >= HGCalTriggerCell) && (triggerCellV() >= HGCalTriggerCell)) {
    int u0 = nT * triggerCellU();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        uc.emplace_back(u0 + i);
      }
    }
  } else if ((triggerCellU() < HGCalTriggerCell) && (triggerCellU() <= triggerCellV())) {
    int u0 = nT * triggerCellU();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        uc.emplace_back(u0 + i);
      }
    }
  } else {
    int u0 = nT * (triggerCellU() - 1) + 1;
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        uc.emplace_back(u0 + j);
      }
      ++u0;
    }
  }
  return uc;
}

std::vector<int> HGCalTriggerDetId::cellV() const {
  std::vector<int> vc;
  int nT =
      (type() == HGCSiliconDetId::HGCalFine) ? HGCSiliconDetId::HGCalFineTrigger : HGCSiliconDetId::HGCalCoarseTrigger;
  if ((triggerCellU() >= HGCalTriggerCell) && (triggerCellV() >= HGCalTriggerCell)) {
    int v0 = nT * triggerCellV();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        vc.emplace_back(v0 + j);
      }
    }
  } else if ((triggerCellU() < HGCalTriggerCell) && (triggerCellU() <= triggerCellV())) {
    int v0 = nT * triggerCellV();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        vc.emplace_back(v0 + j);
      }
      ++v0;
    }
  } else {
    int v0 = nT * triggerCellV();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        vc.emplace_back(v0 + i);
      }
    }
  }
  return vc;
}

std::vector<std::pair<int, int> > HGCalTriggerDetId::cellUV() const {
  std::vector<int> uc = cellU();
  std::vector<int> vc = cellV();
  std::vector<std::pair<int, int> > uv;
  for (unsigned int k = 0; k < uc.size(); ++k) {
    uv.emplace_back(std::pair<int, int>(uc[k], vc[k]));
  }
  return uv;
}

std::ostream& operator<<(std::ostream& s, const HGCalTriggerDetId& id) {
  return s << " EE:HSil= " << id.isEE() << ":" << id.isHSilicon() << " type= " << id.type() << " z= " << id.zside()
           << " layer= " << id.layer() << " wafer(u,v:x,y)= (" << id.waferU() << "," << id.waferV() << ":"
           << id.waferX() << "," << id.waferY() << ")"
           << " triggerCell(u,v:x,y)= (" << id.triggerCellU() << "," << id.triggerCellV() << ":" << id.triggerCellX()
           << "," << id.triggerCellY() << ")";
}
