/** \file
 * 
 *  $Date: 2014/02/02 22:12:32 $
 *  $Revision: 1.0 $
 *
 * \author Marcello Maggi
 */


#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include <iostream>

ME0DigiPreReco::ME0DigiPreReco (float x, float y, float ex, float ey, float corr, float tof) :
  x_(x),
  y_(y),
  ex_(ex),
  ey_(ey),
  corr_(corr),
  tof_(tof)
{}

ME0DigiPreReco::ME0DigiPreReco ():
  x_(0.),
  y_(0.),
  ex_(0.),
  ey_(0.),
  corr_(0.),
  tof_(-1.)
{}


// Comparison
bool ME0DigiPreReco::operator == (const ME0DigiPreReco& digi) const
{
  if ( x_ != digi.x() ||
       y_ != digi.y() || 
       tof_ != digi.tof()
       ) return false;
  return true;
}


///Precedence operator
bool ME0DigiPreReco::operator<(const ME0DigiPreReco& digi) const
{
  if (digi.tof() == tof_){
    if(digi.x() == x_)
      return digi.y() < y_;
    else 
      return digi.x() < x_;
  } else {
    return digi.tof() < tof_;
  }
}


std::ostream & operator<<(std::ostream & o, const ME0DigiPreReco& digi)
{
  return o << "local x=" << digi.x() << " cm y=" << digi.y()<<" cm ex=" << digi.ex() << " cm ey=" << digi.ey()<< " cm tof="<<digi.tof()<<" ns";
}

void ME0DigiPreReco::print() const
{
  std::cout << "local x=" << this->x() << " cm y=" << this->y() <<" cm tof="<<this->tof()<<" ns"<<std::endl;
}

