#include "DataFormats/GEMDigi/interface/ME0PadDigiCluster.h"
#include <iostream>

ME0PadDigiCluster::ME0PadDigiCluster(std::vector<uint16_t> pads, int bx) : v_(pads), bx_(bx) {}

ME0PadDigiCluster::ME0PadDigiCluster() : v_(std::vector<uint16_t>()), bx_(0) {}

// Comparison
bool ME0PadDigiCluster::operator==(const ME0PadDigiCluster& digi) const {
  return v_ == digi.pads() and bx_ == digi.bx();
}

// Comparison
bool ME0PadDigiCluster::operator!=(const ME0PadDigiCluster& digi) const {
  return v_ != digi.pads() or bx_ != digi.bx();
}

///Precedence operator
bool ME0PadDigiCluster::operator<(const ME0PadDigiCluster& digi) const {
  if (digi.bx() == bx_)
    return digi.pads().front() < v_.front();
  else
    return digi.bx() < bx_;
}

std::ostream& operator<<(std::ostream& o, const ME0PadDigiCluster& digi) {
  o << " bx: " << digi.bx() << " pads: ";
  for (auto p : digi.pads())
    o << " " << p;
  o << std::endl;
  return o;
}

void ME0PadDigiCluster::print() const {
  std::cout << " bx: " << bx() << " pads: ";
  for (auto p : pads())
    std::cout << " " << p;
  std::cout << std::endl;
}
