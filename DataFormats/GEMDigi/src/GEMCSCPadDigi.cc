/** \file
 * 
 *  $Date: 2013/01/18 04:21:50 $
 *  $Revision: 1.1 $
 *
 * \author Vadim Khotilovich
 */


#include "DataFormats/GEMDigi/interface/GEMCSCPadDigi.h"
#include <iostream>

GEMCSCPadDigi::GEMCSCPadDigi (int pad, int bx) :
  pad_(pad),
  bx_(bx)
{}

GEMCSCPadDigi::GEMCSCPadDigi ():
  pad_(0),
  bx_(0) 
{}


// Comparison
bool GEMCSCPadDigi::operator == (const GEMCSCPadDigi& digi) const
{
  if ( pad_ != digi.pad() ||
       bx_    != digi.bx() ) return false;
  return true;
}


///Precedence operator
bool GEMCSCPadDigi::operator<(const GEMCSCPadDigi& digi) const
{
  if(digi.bx() == bx_)
    return digi.pad() < pad_;
  else 
    return digi.bx() < bx_;
}


std::ostream & operator<<(std::ostream & o, const GEMCSCPadDigi& digi)
{
  return o << " " << digi.pad() << " " << digi.bx();
}


void GEMCSCPadDigi::print() const
{
  std::cout << "Pad " << pad() << " bx " << bx() <<std::endl;
}

