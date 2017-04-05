#include "DataFormats/GEMDigi/interface/ME0Digi.h"
#include <iostream>

ME0Digi::ME0Digi (int strip, int bx) :
  strip_(strip),
  bx_(bx)
{}

ME0Digi::ME0Digi ():
  strip_(0),
  bx_(0) 
{}


// Comparison
bool ME0Digi::operator == (const ME0Digi& digi) const
{
  return strip_ == digi.strip() and bx_ == digi.bx();
}


// Comparison
bool ME0Digi::operator != (const ME0Digi& digi) const
{
  return strip_ != digi.strip() or bx_ != digi.bx();
}


///Precedence operator
bool ME0Digi::operator<(const ME0Digi& digi) const
{
  if(digi.bx() == bx_)
    return digi.strip() < strip_;
  else 
    return digi.bx() < bx_;
}


std::ostream & operator<<(std::ostream & o, const ME0Digi& digi)
{
  return o << " " << digi.strip() << " " << digi.bx();
}


void ME0Digi::print() const
{
  std::cout << "Strip " << strip() << " bx " << bx() <<std::endl;
}

