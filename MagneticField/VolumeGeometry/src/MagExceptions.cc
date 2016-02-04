#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

MagException::MagException(const char *message) : theMessage(message) {}
MagException::~MagException() throw() {}
const char* 
MagException::what() const throw() { return theMessage.c_str();}

GridInterpolator3DException::GridInterpolator3DException(double a1, double b1, double c1,
                                                         double a2, double b2, double c2)  throw() 
{
  limits_[0] = a1;
  limits_[1] = b1;
  limits_[2] = c1;
  limits_[3] = a2;
  limits_[4] = b2;
  limits_[5] = c2;
}

GridInterpolator3DException::~GridInterpolator3DException() throw() {}

const char* 
GridInterpolator3DException::what() const throw() { return "LinearGridInterpolator3D: field requested outside of grid validity";}
