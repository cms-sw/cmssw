#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HFNoseTriggerDetId HFNoseTriggerDetId::Undefined(HFNoseTrigger, 0, 0, 0, 0, 0, 0, 0);

HFNoseTriggerDetId::HFNoseTriggerDetId() : DetId() {}

HFNoseTriggerDetId::HFNoseTriggerDetId(uint32_t rawid) : DetId(rawid) {}

HFNoseTriggerDetId::HFNoseTriggerDetId(
    int subdet, int zp, int type, int layer, int waferU, int waferV, int cellU, int cellV)
    : DetId(HGCalTrigger, ForwardEmpty) {
  int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int waferUsign = (waferU >= 0) ? 0 : 1;
  int waferVsign = (waferV >= 0) ? 0 : 1;
  int zside = (zp < 0) ? 1 : 0;
  id_ |= (((cellU & kHFNoseCellUMask) << kHFNoseCellUOffset) | ((cellV & kHFNoseCellVMask) << kHFNoseCellVOffset) |
          ((waferUabs & kHFNoseWaferUMask) << kHFNoseWaferUOffset) |
          ((waferUsign & kHFNoseWaferUSignMask) << kHFNoseWaferUSignOffset) |
          ((waferVabs & kHFNoseWaferVMask) << kHFNoseWaferVOffset) |
          ((waferVsign & kHFNoseWaferVSignMask) << kHFNoseWaferVSignOffset) |
          ((layer & kHFNoseLayerMask) << kHFNoseLayerOffset) | ((zside & kHFNoseZsideMask) << kHFNoseZsideOffset) |
          ((type & kHFNoseTypeMask) << kHFNoseTypeOffset) | ((subdet & kHFNoseSubdetMask) << kHFNoseSubdetOffset));
}

HFNoseTriggerDetId::HFNoseTriggerDetId(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det() != HGCalTrigger) || ((gen.subdetId() & kHFNoseSubdetMask) != HFNoseTrigger)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HFNoseTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HFNoseTriggerDetId& HFNoseTriggerDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det() != HGCalTrigger) || ((gen.subdetId() & kHFNoseSubdetMask) != HFNoseTrigger)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HFNoseTriggerDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

int HFNoseTriggerDetId::triggerCellX() const {
  int nT = (type() == HFNoseDetId::HFNoseFine) ? HFNoseDetId::HFNoseFineTrigger : HFNoseDetId::HFNoseCoarseTrigger;
  int N = nT * HFNoseTriggerCell;
  std::vector<int> vc = cellV();
  int x(0);
  for (auto const& v : vc) {
    x += (3 * (v - N) + 2);
  }
  return (x / static_cast<int>(vc.size()));
}

int HFNoseTriggerDetId::triggerCellY() const {
  int nT = (type() == HFNoseDetId::HFNoseFine) ? HFNoseDetId::HFNoseFineTrigger : HFNoseDetId::HFNoseCoarseTrigger;
  int N = nT * HFNoseTriggerCell;
  std::vector<int> uc = cellU();
  std::vector<int> vc = cellV();
  int y(0);
  for (unsigned int k = 0; k < uc.size(); ++k) {
    y += (2 * uc[k] - (N + vc[k]));
  }
  return (y / static_cast<int>(vc.size()));
}

std::vector<int> HFNoseTriggerDetId::cellU() const {
  std::vector<int> uc;
  int nT = (type() == HFNoseDetId::HFNoseFine) ? HFNoseDetId::HFNoseFineTrigger : HFNoseDetId::HFNoseCoarseTrigger;
  if ((triggerCellU() >= HFNoseTriggerCell) && (triggerCellV() >= HFNoseTriggerCell)) {
    int u0 = nT * triggerCellU();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        uc.emplace_back(u0 + i);
      }
    }
  } else if ((triggerCellU() < HFNoseTriggerCell) && (triggerCellU() <= triggerCellV())) {
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

std::vector<int> HFNoseTriggerDetId::cellV() const {
  std::vector<int> vc;
  int nT = (type() == HFNoseDetId::HFNoseFine) ? HFNoseDetId::HFNoseFineTrigger : HFNoseDetId::HFNoseCoarseTrigger;
  if ((triggerCellU() >= HFNoseTriggerCell) && (triggerCellV() >= HFNoseTriggerCell)) {
    int v0 = nT * triggerCellV();
    for (int i = 0; i < nT; ++i) {
      for (int j = 0; j < nT; ++j) {
        vc.emplace_back(v0 + j);
      }
    }
  } else if ((triggerCellU() < HFNoseTriggerCell) && (triggerCellU() <= triggerCellV())) {
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

std::vector<std::pair<int, int> > HFNoseTriggerDetId::cellUV() const {
  std::vector<int> uc = cellU();
  std::vector<int> vc = cellV();
  std::vector<std::pair<int, int> > uv;
  for (unsigned int k = 0; k < uc.size(); ++k) {
    uv.emplace_back(uc[k], vc[k]);
  }
  return uv;
}

std::ostream& operator<<(std::ostream& s, const HFNoseTriggerDetId& id) {
  return s << " EE:HSil= " << id.isEE() << ":" << id.isHSilicon() << " type= " << id.type() << " z= " << id.zside()
           << " layer= " << id.layer() << " wafer(u,v:x,y)= (" << id.waferU() << "," << id.waferV() << ":"
           << id.waferX() << "," << id.waferY() << ")"
           << " triggerCell(u,v:x,y)= (" << id.triggerCellU() << "," << id.triggerCellV() << ":" << id.triggerCellX()
           << "," << id.triggerCellY() << ")";
}
