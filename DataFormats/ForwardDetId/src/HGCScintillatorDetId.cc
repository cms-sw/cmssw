#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCScintillatorDetId HGCScintillatorDetId::Undefined(0, 0, 0, 0, false);

HGCScintillatorDetId::HGCScintillatorDetId() : DetId() {}

HGCScintillatorDetId::HGCScintillatorDetId(uint32_t rawid) : DetId(rawid) {}

HGCScintillatorDetId::HGCScintillatorDetId(int type, int layer, int radius, int phi, bool trigger, int sipm)
    : DetId(HGCalHSc, ForwardEmpty) {
  int zside = (radius < 0) ? 1 : 0;
  int itrig = trigger ? 1 : 0;
  int radiusAbs = std::abs(radius);
  id_ |= (((type & kHGCalTypeMask) << kHGCalTypeOffset) | ((zside & kHGCalZsideMask) << kHGCalZsideOffset) |
          ((sipm & kHGCalSiPMMask) << kHGCalSiPMOffset) | ((itrig & kHGCalTriggerMask) << kHGCalTriggerOffset) |
          ((layer & kHGCalLayerMask) << kHGCalLayerOffset) | ((radiusAbs & kHGCalRadiusMask) << kHGCalRadiusOffset) |
          ((phi & kHGCalPhiMask) << kHGCalPhiOffset));
}

HGCScintillatorDetId::HGCScintillatorDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != HGCalHSc) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
}

HGCScintillatorDetId& HGCScintillatorDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det() != HGCalHSc) {
      throw cms::Exception("Invalid DetId")
          << "Cannot assign HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
    }
  }
  id_ = gen.rawId();
  return (*this);
}

int HGCScintillatorDetId::iradiusAbs() const {
  if (trigger())
    return (2 * ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask));
  else
    return ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask);
}

int HGCScintillatorDetId::iradiusTriggerAbs() const {
  if (trigger())
    return ((iradiusAbs() + 1) / 2);
  else
    return iradiusAbs();
}

int HGCScintillatorDetId::iphi() const {
  if (trigger())
    return (2 * ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask));
  else
    return ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask);
}

int HGCScintillatorDetId::iphiTrigger() const {
  if (trigger())
    return ((iphi() + 1) / 2);
  else
    return iphi();
}

void HGCScintillatorDetId::setType(int type) {
  id_ &= kHGCalTypeMask0;
  id_ |= ((type & kHGCalTypeMask) << kHGCalTypeOffset);
}

void HGCScintillatorDetId::setSiPM(int sipm) {
  id_ &= kHGCalSiPMMask0;
  id_ |= ((sipm & kHGCalSiPMMask) << kHGCalSiPMOffset);
}

std::vector<HGCScintillatorDetId> HGCScintillatorDetId::detectorCells() const {
  std::vector<HGCScintillatorDetId> cells;
  int irad = iradiusAbs();
  int ifi = iphi();
  int iz = zside();
  if (trigger()) {
    cells.emplace_back(HGCScintillatorDetId(type(), layer(), (2 * irad - 1) * iz, 2 * ifi - 1, false));
    cells.emplace_back(HGCScintillatorDetId(type(), layer(), 2 * irad * iz, 2 * ifi - 1, false));
    cells.emplace_back(HGCScintillatorDetId(type(), layer(), (2 * irad - 1) * iz, 2 * ifi, false));
    cells.emplace_back(HGCScintillatorDetId(type(), layer(), 2 * irad * iz, 2 * ifi, false));
  } else {
    cells.emplace_back(HGCScintillatorDetId(type(), layer(), irad * iz, ifi, false));
  }
  return cells;
}

HGCScintillatorDetId HGCScintillatorDetId::geometryCell() const {
  if (trigger()) {
    return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), false);
  } else {
    return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), false);
  }
}

HGCScintillatorDetId HGCScintillatorDetId::triggerCell() const {
  if (trigger())
    return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), true);
  else
    return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), true);
}

std::ostream& operator<<(std::ostream& s, const HGCScintillatorDetId& id) {
  return s << " HGCScintillatorDetId::EE:HE= " << id.isEE() << ":" << id.isHE() << " trigger= " << id.trigger()
           << " type= " << id.type() << " SiPM= " << id.sipm() << " layer= " << id.layer()
           << " radius= " << id.iradius() << ":" << id.iradiusTrigger() << " phi= " << id.iphi() << ":"
           << id.iphiTrigger();
}
