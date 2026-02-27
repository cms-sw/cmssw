/** \file
 * 
 *
 * \author Borislav Pavlov - University of Sofia
 *
 */

#include "DataFormats/RPCDigi/interface/RPCDigiPhase2.h"
#include <iostream>

RPCDigiPhase2::RPCDigiPhase2(int strip, int bx, int sbx) : strip_(strip), bx_(bx), sbx_(sbx) {}

RPCDigiPhase2::RPCDigiPhase2() : strip_(0), bx_(0), sbx_(0) {}

// Comparison
bool RPCDigiPhase2::operator==(const RPCDigiPhase2& digi) const {
  if (strip_ != digi.strip() || bx_ != digi.bx())
    return false;
  return true;
}

///Precedence operator
bool RPCDigiPhase2::operator<(const RPCDigiPhase2& digi) const {
  if (digi.bx() == this->bx())
    return digi.strip() < this->strip();
  else
    return digi.bx() < this->bx();
}

std::ostream& operator<<(std::ostream& o, const RPCDigiPhase2& digi) {
  return o << " " << digi.strip() << " " << digi.bx();
}

void RPCDigiPhase2::print() const { std::cout << "Strip " << strip() << " bx " << bx() << std::endl; }
