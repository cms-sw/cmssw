/** \file
 * 
 *
 * \author Ilaria Segoni
 *
 * modified by Borislav Pavlov - University of Sofia
 * modification to be used for upgrade and for "pseudodigi"
 *
 *
 */

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include <iostream>

RPCDigi::RPCDigi(int strip, int bx)
    : strip_(strip),
      bx_(bx),
      time_(0),
      coordinateX_(0),
      coordinateY_(0),
      deltaTime_(0),
      deltaX_(0),
      deltaY_(0),
      hasTime_(false),
      hasX_(false),
      hasY_(false) {}

RPCDigi::RPCDigi()
    : strip_(0),
      bx_(0),
      time_(0),
      coordinateX_(0),
      coordinateY_(0),
      deltaTime_(0),
      deltaX_(0),
      deltaY_(0),
      hasTime_(false),
      hasX_(false),
      hasY_(false) {}

// Comparison
bool RPCDigi::operator==(const RPCDigi& digi) const {
  if (strip_ != digi.strip() || bx_ != digi.bx())
    return false;
  return true;
}

///Precedence operator
bool RPCDigi::operator<(const RPCDigi& digi) const {
  if (digi.bx() == this->bx())
    return digi.strip() < this->strip();
  else
    return digi.bx() < this->bx();
}

std::ostream& operator<<(std::ostream& o, const RPCDigi& digi) { return o << " " << digi.strip() << " " << digi.bx(); }

void RPCDigi::print() const { std::cout << "Strip " << strip() << " bx " << bx() << std::endl; }
