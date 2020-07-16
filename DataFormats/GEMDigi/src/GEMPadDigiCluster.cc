#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include <iostream>

GEMPadDigiCluster::GEMPadDigiCluster(std::vector<uint16_t> pads, int16_t bx, enum GEMSubDetId::Station station)
    : v_(pads), bx_(bx), station_(station) {}

GEMPadDigiCluster::GEMPadDigiCluster() : v_(std::vector<uint16_t>()), bx_(-99), station_(GEMSubDetId::Station::GE11) {}

// Comparison
bool GEMPadDigiCluster::operator==(const GEMPadDigiCluster& digi) const {
  return v_ == digi.pads() and bx_ == digi.bx() and station_ == digi.station();
}

// Comparison
bool GEMPadDigiCluster::operator!=(const GEMPadDigiCluster& digi) const {
  return v_ != digi.pads() or bx_ != digi.bx();
}

///Precedence operator
bool GEMPadDigiCluster::operator<(const GEMPadDigiCluster& digi) const {
  if (digi.bx() == bx_)
    return digi.pads().front() < v_.front();
  else
    return digi.bx() < bx_;
}

bool GEMPadDigiCluster::isValid() const {
  // empty clusters are always invalid
  if (v_.empty())
    return false;

  uint16_t invalid = GE11InValid;
  if (station_ == GEMSubDetId::Station::GE21) {
    invalid = GE21InValid;
  }
  return v_[0] != invalid;
}

std::ostream& operator<<(std::ostream& o, const GEMPadDigiCluster& digi) {
  o << " bx: " << digi.bx() << " pads: ";
  for (auto p : digi.pads())
    o << " " << p;
  o << std::endl;
  return o;
}

void GEMPadDigiCluster::print() const {
  std::cout << " bx: " << bx() << " pads: ";
  for (auto p : pads())
    std::cout << " " << p;
  std::cout << std::endl;
}
