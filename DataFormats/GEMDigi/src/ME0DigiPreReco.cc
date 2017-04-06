/** \file
 * 
 *  $Date: 2014/02/02 22:12:32 $
 *  $Revision: 1.0 $
 *
 * \author Marcello Maggi
 */


#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include <iostream>

//ME0DigiPreReco::ME0DigiPreReco (float x, float y, float ex, float ey, float corr, float tof) :
ME0DigiPreReco::ME0DigiPreReco (float x, float y, float ex, float ey, float corr, float tof, int pdgid, int prompt) :
  x_(x),
  y_(y),
  ex_(ex),
  ey_(ey),
  corr_(corr),
  tof_(tof),
  pdgid_(pdgid),
  prompt_(prompt)
{}

ME0DigiPreReco::ME0DigiPreReco ():
  x_(0.),
  y_(0.),
  ex_(0.),
  ey_(0.),
  corr_(0.),
  tof_(-1.),
  pdgid_(0),
  prompt_(0)
{}


// Comparison
bool ME0DigiPreReco::operator == (const ME0DigiPreReco& digi) const
{
  return x_ == digi.x() and y_ == digi.y() and tof_ == digi.tof();
}


// Comparison
bool ME0DigiPreReco::operator != (const ME0DigiPreReco& digi) const
{
  return x_ != digi.x() or y_ != digi.y() or tof_ != digi.tof();
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
//  return o << "local x=" << digi.x() << " cm y=" << digi.y()<<" cm ex=" << digi.ex() << " cm ey=" << digi.ey()<< " cm tof="<<digi.tof()<<" ns";
  return o << "local x=" << digi.x() << " cm y=" << digi.y()<<" cm ex=" << digi.ex() << " cm ey=" << digi.ey()<< " cm tof="<<digi.tof()<<" ns"<<" pdgID "<<digi.pdgid()<<" prompt? "<<digi.prompt();
}

void ME0DigiPreReco::print() const
{
//  std::cout << "local x=" << this->x() << " cm y=" << this->y() <<" cm tof="<<this->tof()<<" ns"<<std::endl;
  std::cout << "local x=" << this->x() << " cm y=" << this->y() <<" cm tof="<<this->tof()<<" ns"<<" pdgID "<<this->pdgid()<<" prompt? "<<this->prompt()<<std::endl;
}

