#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include <iostream>

GEMPadDigiCluster::GEMPadDigiCluster (int firstPad, int lastPad, int bx) :
  firstPad_(firstPad),
  lastPad_(lastPad),
  bx_(bx)
{}

GEMPadDigiCluster::GEMPadDigiCluster ():
  firstPad_(0),
  lastPad_(0),
  bx_(0) 
{}


// Comparison
bool GEMPadDigiCluster::operator == (const GEMPadDigiCluster& digi) const
{
  return firstPad_ == digi.firstPad() and lastPad_ == digi.lastPad() and bx_ == digi.bx();
}


// Comparison
bool GEMPadDigiCluster::operator != (const GEMPadDigiCluster& digi) const
{
  return firstPad_ != digi.firstPad() or lastPad_ != digi.lastPad() or bx_ != digi.bx();
}


///Precedence operator
bool GEMPadDigiCluster::operator<(const GEMPadDigiCluster& digi) const
{
  if(digi.bx() == bx_)
    return digi.firstPad() < firstPad_;
  else 
    return digi.bx() < bx_;
}


std::ostream & operator<<(std::ostream & o, const GEMPadDigiCluster& digi)
{
  return o << " " << digi.firstPad() << " " << digi.lastPad() <<" " << digi.bx();
}


void GEMPadDigiCluster::print() const
{
  std::cout << "First pad " << firstPad() << "Last pad " << lastPad() <<" bx " << bx() <<std::endl;
}

