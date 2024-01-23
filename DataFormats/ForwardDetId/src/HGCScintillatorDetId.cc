#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include <ostream>
#include <iostream>

const HGCScintillatorDetId HGCScintillatorDetId::Undefined(0, 0, 0, 0, false);

std::vector<HGCScintillatorDetId> HGCScintillatorDetId::detectorCells() const {
  std::vector<HGCScintillatorDetId> cells;
  int irad = ring();
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

std::ostream& operator<<(std::ostream& s, const HGCScintillatorDetId& id) {
  return s << " HGCScintillatorDetId::EE:HE= " << id.isEE() << ":" << id.isHE() << " trigger= " << id.trigger()
           << " type= " << id.type() << " SiPM= " << id.sipm() << " layer= " << id.layer() << " ring= " << id.iradius()
           << ":" << id.iradiusTrigger() << " phi= " << id.iphi() << ":" << id.iphiTrigger();
}
