/** \file
 * 
 *  $Date: 2006/05/16 14:36:29 $
 *  $Revision: 1.7 $
 *
 * \author Ilaria Segoni
 */


#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include <iostream>

RPCDigi::RPCDigi (int strip, int bx) :
  strip_(strip),
  bx_(bx)
{}

RPCDigi::RPCDigi ():
  strip_(0),
  bx_(0) 
{}


// Comparison
bool
RPCDigi::operator == (const RPCDigi& digi) const {
  if ( strip_ != digi.strip() ||
       bx_    != digi.bx() ) return false;
  return true;
}

///Precedence operator
bool 
RPCDigi::operator<(const RPCDigi& digi) const{

  if(digi.bx() == this->bx())
    return digi.strip()<this->strip();
  else 
    return digi.bx()<this->bx();
}

std::ostream & operator<<(std::ostream & o, const RPCDigi& digi) {
  return o << " " << digi.strip()
           << " " << digi.bx();
}

int RPCDigi::strip() const { return strip_; }

int RPCDigi::bx() const { return bx_; }

void
RPCDigi::print() const {
  std::cout << "Strip " << strip() 
       << " bx " << bx() <<std::endl;
}

