/** \file
 * 
 *
 * \author Borislav Pavlov - University of Sofia
 *
 */

#include "DataFormats/RPCDigi/interface/IRPCDigi.h"
#include <iostream>

IRPCDigi::IRPCDigi(int strip, int bxLR, int bxHR, int sbxLR, int sbxHR, int fineLR, int fineHR)
    : strip_(strip), bxLR_(bxLR), bxHR_(bxHR), sbxLR_(sbxLR), sbxHR_(sbxHR), fineLR_(fineLR), fineHR_(fineHR) {}

IRPCDigi::IRPCDigi() : strip_(0), bxLR_(0), sbxLR_(0) {}

// Comparison
bool IRPCDigi::operator==(const IRPCDigi& digi) const {
  if (strip_ != digi.strip() || bxLR_ != digi.bx())
    return false;
  return true;
}

///Precedence operator
bool IRPCDigi::operator<(const IRPCDigi& digi) const {
  if (digi.bx() == this->bx())
    return digi.strip() < this->strip();
  else
    return digi.bx() < this->bx();
}

std::ostream& operator<<(std::ostream& o, const IRPCDigi& digi) { return o << " " << digi.strip() << " " << digi.bx(); }

void IRPCDigi::print() const { std::cout << "Strip " << strip() << " bx " << bx() << std::endl; }
