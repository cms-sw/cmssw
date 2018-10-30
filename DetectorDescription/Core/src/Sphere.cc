#include "DetectorDescription/Core/src/Sphere.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <cmath>
#include <ostream>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

using namespace dd;
using namespace dd::operators;

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
  os << " innerRadius=" << CONVERT_TO( p_[0], cm )
     << " outerRadius=" << CONVERT_TO( p_[1], cm )
     << " startPhi=" << CONVERT_TO( p_[2], deg )
     << " deltaPhi=" << CONVERT_TO( p_[3], deg )
     << " startTheta=" << CONVERT_TO( p_[4], deg )
     << " deltaTheta=" << CONVERT_TO( p_[5], deg );
}

double DDI::Sphere::volume() const
{
  double volume(0.);
  if ( std::fabs(p_[3]) <= 2._pi && std::fabs(p_[5]) <= _pi ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]) - std::cos(p_[4]))*p_[3]);
  } else if (std::fabs(p_[3]) <= 2._pi && std::fabs(p_[5]) > _pi ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]-180._deg) - std::cos(p_[4]))*p_[3]);
  } else if (std::fabs(p_[3]) > 2._pi && std::fabs(p_[5]) <= _pi ) {
    volume = std::fabs((p_[1]*p_[1]*p_[1] - p_[0]*p_[0]*p_[0])/3. * (std::cos(p_[4]+p_[5]) - std::cos(p_[4]))*(p_[3]-p_[2]));
  }
  return volume;
}
