#include "DetectorDescription/Core/src/Sphere.h"
#include <DataFormats/GeometryVector/interface/Pi.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cmath>
#include <ostream>

DDI::Sphere::Sphere(double innerRadius,
		    double outerRadius,
		    double startPhi,
		    double deltaPhi,
		    double startTheta,
		    double deltaTheta)
 : Solid(ddsphere)	  
{
  p_.push_back(innerRadius);
  p_.push_back(outerRadius);
  p_.push_back(startPhi);
  p_.push_back(deltaPhi);
  p_.push_back(startTheta);
  p_.push_back(deltaTheta);
}	 


void DDI::Sphere::stream(std::ostream & os) const
{
   os << " innerRadius=" << p_[0]/cm
      << " outerRadius=" << p_[1]/cm
      << " startPhi=" << p_[2]/deg
      << " deltaPhi=" << p_[3]/deg
      << " startTheta=" << p_[4]/deg
      << " deltaTheta=" << p_[5]/deg;
}

double DDI::Sphere::volume() const
{
  double volume(0.);
  if ( std::fabs(p_[3]) <= 2.*Geom::pi() && std::fabs(p_[5]) <= Geom::pi() ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]) - std::cos(p_[4]))*p_[3]);
  } else if (std::fabs(p_[3]) <= 2.*Geom::pi() && std::fabs(p_[5]) > Geom::pi() ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]-180.*deg) - std::cos(p_[4]))*p_[3]);
  } else if (std::fabs(p_[3]) > 2.*Geom::pi() && std::fabs(p_[5]) <= Geom::pi() ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]) - std::cos(p_[4]))*(p_[3]-p_[2]));
  }
  return volume;
}


