#include "DetectorDescription/Core/src/Sphere.h"

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <cmath>
#include <ostream>
#include <vector>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

DDI::Sphere::Sphere(double innerRadius,
		    double outerRadius,
		    double startPhi,
		    double deltaPhi,
		    double startTheta,
		    double deltaTheta)
  : Solid(DDSolidShape::ddsphere)	  
{
  p_.emplace_back(innerRadius);
  p_.emplace_back(outerRadius);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  p_.emplace_back(startTheta);
  p_.emplace_back(deltaTheta);
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


