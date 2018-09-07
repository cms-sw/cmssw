#include "DetectorDescription/Core/src/Trap.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

using std::sqrt;
using namespace dd::operators;

DDI::Trap::Trap( double pDz, 
                 double pTheta,
                 double pPhi,
                 double pDy1, double pDx1,double pDx2,
                 double pAlp1,
                 double pDy2, double pDx3, double pDx4,
                 double pAlp2 )
 : Solid(DDSolidShape::ddtrap) 
{		 
  p_.emplace_back(pDz); // ......... 0
  p_.emplace_back(pTheta); // .. 1
  p_.emplace_back(pPhi); // ....... 2
  p_.emplace_back(pDy1); // ........ 3
  p_.emplace_back(pDx1); // ........ 4
  p_.emplace_back(pDx2); // ........ 5
  p_.emplace_back(pAlp1); // ....... 6
  p_.emplace_back(pDy2); // ........ 7
  p_.emplace_back(pDx3); // ......... 8
  p_.emplace_back(pDx4); // ........ 9
  p_.emplace_back(pAlp2);
}


void DDI::Trap::stream(std::ostream & os) const
{
  os << " dz=" << CONVERT_TO( p_[0], cm )
     << " theta=" << CONVERT_TO( p_[1], deg )
     << " phi=" << CONVERT_TO( p_[2], deg )
     << " dy1=" << CONVERT_TO( p_[3], cm )
     << " dx1=" << CONVERT_TO( p_[4], cm )
     << " dx2=" << CONVERT_TO( p_[5], cm )
     << " alpha1=" << CONVERT_TO( p_[6], deg )
     << " dy2=" << CONVERT_TO( p_[7], cm )
     << " dx3=" << CONVERT_TO( p_[8], cm )
     << " dx4=" << CONVERT_TO( p_[9], cm )
     << " alpha2=" << CONVERT_TO( p_[10], deg );
}

double DDI::Trap::volume() const
{
  double volume = 0;

  double dz  = p_[0]*2.;
  double dy1 = p_[3]*2.;
  double dx1 = p_[4]*2.;
  double dx2 = p_[5]*2.;
  double dy2 = p_[7]*2.;
  double dx3 = p_[8]*2.;
  double dx4 = p_[9]*2.;

  volume = ((dx1 + dx2 + dx3 + dx4)*(dy1 + dy2) +
	    (dx4 + dx3 - dx2 - dx1)*(dy2 - dy1)/3)*dz*0.125;

  return volume;
}
