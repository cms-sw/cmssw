/** \file
 * 
 * \author Sven Dildick
 */


#include "DataFormats/GEMDigi/interface/ME0PadDigi.h"
#include <iostream>

ME0PadDigi::ME0PadDigi (int pad, int bx) :
  pad_(pad),
  bx_(bx)
{}

ME0PadDigi::ME0PadDigi ():
  pad_(0),
  bx_(0) 
{}


// Comparison
bool ME0PadDigi::operator == (const ME0PadDigi& digi) const
{
  return pad_ == digi.pad() and bx_ == digi.bx();
}


// Comparison
bool ME0PadDigi::operator != (const ME0PadDigi& digi) const
{
  return pad_ != digi.pad() or bx_ != digi.bx();
}


///Precedence operator
bool ME0PadDigi::operator<(const ME0PadDigi& digi) const
{
  if(digi.bx() == bx_)
    return digi.pad() < pad_;
  else 
    return digi.bx() < bx_;
}


std::ostream & operator<<(std::ostream & o, const ME0PadDigi& digi)
{
  return o << " " << digi.pad() << " " << digi.bx();
}


void ME0PadDigi::print() const
{
  std::cout << "Pad " << pad() << " bx " << bx() <<std::endl;
}

