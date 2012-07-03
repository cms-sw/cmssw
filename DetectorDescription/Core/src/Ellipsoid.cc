#include "DetectorDescription/Core/src/Ellipsoid.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <cmath>

void DDI::Ellipsoid::stream(std::ostream & os) const
{
   os << " xSemiAxis[cm]=" << p_[0]/cm
      << " ySemiAxis[cm]=" << p_[1]/cm
      << " zSemiAxis" << p_[2]/cm
      << " zBottomCut" << p_[3]/cm
      << " zTopCut" << p_[4]/cm;
}

double DDI::Ellipsoid::halfVol(double dz, double maxz)  const {
  double volume(0.);
  double z(0.);
  double x;
  double y;
  double c2 = p_[2] * p_[2];
  for (;z <= maxz; z=z+dz) {
    // what is x here? what is y?  This assumes a TRIANGLE approximation 
    // This assumes that I can estimate the decrease in x and y at current z
    // at current z, x is the projection down to the x axis.
    // x = a * sqrt(1-z^2 / c^2 )
    // unneccesary paranoia?
    if ( z*z / c2 >  1.0 ) {
      std::cout << "error!" << std::endl;
      exit(0);
    }
    x = p_[0] * sqrt( 1 - z*z/c2);
    y = p_[1] * sqrt( 1 - z*z/c2);
//     if (dispcount < 100)
//     std::cout << "x = " << x << " y = " << y << " z = " << z << std::endl;
//     ++dispcount;
    volume = volume + Geom::pi() * dz * x  * y;
  }
//   std::cout << " x = " << x;
//   std::cout << " y = " << y;
//   std::cout << " z = " << z;
//   std::cout << " vol = " << volume << std::endl;
  return volume;
}

double DDI::Ellipsoid::volume() const { 
  double volume(0.);
  // default if both p_[3] and p_[4] are 0
  volume = 4./3. * Geom::pi() * p_[0] * p_[1] * p_[2];
  if ( p_[3] > 0.0 ) {
    //fail
    std::cout << "FAIL: p_[3] > 0.0" <<std::endl;
  } else if ( p_[4] < 0.0 ) {
    //fail
    std::cout << "FAIL: p_[4] <  0.0" <<std::endl;
  } else if ( p_[3] < 0. && p_[4] > 0. ) {
    volume = halfVol (p_[4]/100000., p_[4]) + halfVol (std::fabs(p_[3]/100000.), std::fabs(p_[3]));
  } else if ( p_[3] < 0. ) {
    volume = volume / 2 + halfVol(std::fabs(p_[3]/100000.), std::fabs(p_[3]));
  } else if ( p_[4] > 0. ) {
    volume = volume / 2 + halfVol (p_[4]/100000., p_[4]);
  }
  return volume; 

}
