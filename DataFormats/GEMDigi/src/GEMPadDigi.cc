/** \file
 *
 *  $Date: 2013/01/18 04:21:50 $
 *  $Revision: 1.1 $
 *
 * \author Vadim Khotilovich
 */


#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include <iostream>

GEMPadDigi::GEMPadDigi (int pad, int bx) :
  pad_(pad),
  bx_(bx)
{}

GEMPadDigi::GEMPadDigi ():
  pad_(0),
  bx_(0)
{}


// Comparison
bool GEMPadDigi::operator == (const GEMPadDigi& digi) const
{
  return pad_ == digi.pad() and bx_ == digi.bx();
}


// Comparison
bool GEMPadDigi::operator != (const GEMPadDigi& digi) const
{
  return pad_ != digi.pad() or bx_ != digi.bx();
}


///Precedence operator
bool GEMPadDigi::operator<(const GEMPadDigi& digi) const
{
  if(digi.bx() == bx_)
    return digi.pad() < pad_;
  else
    return digi.bx() < bx_;
}


bool GEMPadDigi::isValid() const
{
  return pad_ != 0;
}


std::ostream & operator<<(std::ostream & o, const GEMPadDigi& digi)
{
  return o << " GEMPadDigi Pad = " << digi.pad() << " bx = " << digi.bx();
}


void GEMPadDigi::print() const
{
  std::cout << "Pad " << pad() << " bx " << bx() <<std::endl;
}

