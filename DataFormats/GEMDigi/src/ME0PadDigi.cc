#include "DataFormats/GEMDigi/interface/ME0PadDigi.h"
#include <iostream>

ME0PadDigi::ME0PadDigi(int pad, int bx) : pad_(pad), bx_(bx) {}

ME0PadDigi::ME0PadDigi() : pad_(0), bx_(0) {}

bool ME0PadDigi::operator==(const ME0PadDigi& digi) const { return pad_ == digi.pad() and bx_ == digi.bx(); }

bool ME0PadDigi::operator!=(const ME0PadDigi& digi) const { return pad_ != digi.pad() or bx_ != digi.bx(); }

bool ME0PadDigi::operator<(const ME0PadDigi& digi) const {
  if (digi.bx() == bx_)
    return digi.pad() < pad_;
  else
    return digi.bx() < bx_;
}

std::ostream& operator<<(std::ostream& o, const ME0PadDigi& digi) {
  return o << " pad: " << digi.pad() << " bx: " << digi.bx();
}
