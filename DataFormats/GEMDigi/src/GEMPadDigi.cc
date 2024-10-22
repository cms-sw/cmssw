#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include <iostream>

GEMPadDigi::GEMPadDigi(uint16_t pad, int16_t bx, enum GEMSubDetId::Station station, unsigned nPart)
    : pad_(pad), bx_(bx), station_(station), part_(nPart) {}

GEMPadDigi::GEMPadDigi()
    : pad_(GE11InValid), bx_(-99), station_(GEMSubDetId::Station::GE11), part_(NumberPartitions::GE11) {}

// Comparison
bool GEMPadDigi::operator==(const GEMPadDigi& digi) const {
  return pad_ == digi.pad() and bx_ == digi.bx() and station_ == digi.station();
}

// Comparison
bool GEMPadDigi::operator!=(const GEMPadDigi& digi) const { return pad_ != digi.pad() or bx_ != digi.bx(); }

///Precedence operator
bool GEMPadDigi::operator<(const GEMPadDigi& digi) const {
  if (digi.bx() == bx_)
    return digi.pad() < pad_;
  else
    return digi.bx() < bx_;
}

bool GEMPadDigi::isValid() const {
  uint16_t invalid = GE11InValid;
  if (station_ == GEMSubDetId::Station::ME0) {
    invalid = ME0InValid;
  } else if (station_ == GEMSubDetId::Station::GE21) {
    invalid = GE21InValid;
  }
  return pad_ != invalid;
}

std::ostream& operator<<(std::ostream& o, const GEMPadDigi& digi) {
  return o << " GEMPadDigi Pad = " << digi.pad() << " bx = " << digi.bx();
}

void GEMPadDigi::print() const { std::cout << "Pad " << pad() << " bx " << bx() << std::endl; }
