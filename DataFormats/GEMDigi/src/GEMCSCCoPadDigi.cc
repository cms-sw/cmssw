#include "DataFormats/GEMDigi/interface/GEMCSCCoPadDigi.h"
#include <iostream>

GEMCSCCoPadDigi::GEMCSCCoPadDigi(GEMCSCPadDigi f, GEMCSCPadDigi s):
  first_(f),
  second_(s)
{}


GEMCSCCoPadDigi::GEMCSCCoPadDigi():
  first_(GEMCSCPadDigi()),
  second_(GEMCSCPadDigi())
{}


// Comparison
bool GEMCSCCoPadDigi::operator == (const GEMCSCCoPadDigi& digi) const
{
  return digi.first() == first_ and digi.second() == second_;
}


// Comparison
bool GEMCSCCoPadDigi::operator != (const GEMCSCCoPadDigi& digi) const
{
  return digi.first() != first_ or digi.second() != second_;
}


int GEMCSCCoPadDigi::pad(int l) const
{
  if (l==1) return first_.pad();
  else if (l==2) return second_.pad();
  else return -99; // invalid
}


int GEMCSCCoPadDigi::bx(int l) const
{
  if (l==1) return first_.bx();
  else if (l==2) return second_.bx();
  else return -99; // invalid
}


void GEMCSCCoPadDigi::print() const
{
  std::cout << "Pad1 " << first_.pad() << " bx1 " << first_.bx() 
            << ", Pad2 " << second_.pad() << " bx2 " << second_.bx() << std::endl;
}


std::ostream & operator<<(std::ostream & o, const GEMCSCCoPadDigi& digi)
{
  return o << " 1:" << digi.first() << ", 2:" << digi.second();
}
