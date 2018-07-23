/** \file
 * 
 *
 * \author Vadim Khotilovich
 */


#include "DataFormats/GEMDigi/interface/GEMDigi.h"
#include <iostream>

GEMDigi::GEMDigi (int strip, int bx) :
  strip_(strip),
  bx_(bx)
{}

GEMDigi::GEMDigi ():
  strip_(0),
  bx_(0) 
{}


// Comparison
bool GEMDigi::operator == (const GEMDigi& digi) const
{
  return strip_ == digi.strip() and bx_ == digi.bx();
}


// Comparison
bool GEMDigi::operator != (const GEMDigi& digi) const
{
  return strip_ != digi.strip() or bx_ != digi.bx();
}


///Precedence operator
bool GEMDigi::operator<(const GEMDigi& digi) const
{
  if(digi.bx() == bx_)
    return digi.strip() < strip_;
  else 
    return digi.bx() < bx_;
}


std::ostream & operator<<(std::ostream & o, const GEMDigi& digi)
{
  return o << " GEMDigi strip = " << digi.strip() << " bx = " << digi.bx();
}


void GEMDigi::print() const
{
  std::cout << "Strip " << strip() << " bx " << bx() <<std::endl;
}

