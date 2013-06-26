/** \file
 * 
 *  $Date: 2013/01/18 04:18:32 $
 *  $Revision: 1.2 $
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
  if ( strip_ != digi.strip() ||
       bx_    != digi.bx() ) return false;
  return true;
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
  return o << " " << digi.strip() << " " << digi.bx();
}


void GEMDigi::print() const
{
  std::cout << "Strip " << strip() << " bx " << bx() <<std::endl;
}

