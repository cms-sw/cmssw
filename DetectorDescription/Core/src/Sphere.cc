#include "DetectorDescription/Core/src/Sphere.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
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
      << " startPhi=" << p_[2]/cm
      << " deltaPhi=" << p_[3]/cm
      << " startTheta=" << p_[4]/cm
      << " deltaTheta=" << p_[5]/deg;
}

double DDI::Sphere::volume() const
{

  /* I want the integral from x= minX to x = maxX of pi*y^2 */
 
  /* for a "truncated" sphere. rSin(theta) * rSin(theta) * pi is area of a circle.
     integrate from startTheta to startTheta + deltaTheta over theta  ?

  */
  double volume=0.0;

  return volume;

}


