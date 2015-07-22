#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"
#include <iostream>

GEMCoPadDigi::GEMCoPadDigi(uint8_t roll, GEMPadDigi f, GEMPadDigi s):
  roll_(roll),
  first_(f),
  second_(s)
{}


GEMCoPadDigi::GEMCoPadDigi():
  roll_(0),
  first_(GEMPadDigi()),
  second_(GEMPadDigi())
{}


// Comparison
bool GEMCoPadDigi::operator == (const GEMCoPadDigi& digi) const
{
  return digi.first() == first_ and digi.second() == second_ and digi.roll() == roll_;
}


// Comparison
bool GEMCoPadDigi::operator != (const GEMCoPadDigi& digi) const
{
  return digi.first() != first_ or digi.second() != second_ or digi.roll() != roll_;
}


int GEMCoPadDigi::pad(int l) const
{
  if (l==1) return first_.pad();
  else if (l==2) return second_.pad();
  else return -99; // invalid
}


int GEMCoPadDigi::bx(int l) const
{
  if (l==1) return first_.bx();
  else if (l==2) return second_.bx();
  else return -99; // invalid
}


void GEMCoPadDigi::print() const
{
  std::cout << "Roll " << roll_ << ", pad1 " << first_.pad() << " bx1 " << first_.bx() 
            << ", Pad2 " << second_.pad() << " bx2 " << second_.bx() << std::endl;
}


std::ostream & operator<<(std::ostream & o, const GEMCoPadDigi& digi)
{
  return o << "Roll: " << digi.roll() << " 1:" << digi.first() << ", 2:" << digi.second();
}
