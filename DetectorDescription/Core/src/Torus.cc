#include "DetectorDescription/Core/src/Torus.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

using std::sqrt;
using namespace dd;
using namespace dd::operators;

DDI::Torus::Torus( double pRMin,
		   double pRMax,
		   double pRTor,
		   double pSPhi,
		   double pDPhi
		   )
 : Solid(DDSolidShape::ddtorus) 
{		 
  p_.emplace_back(pRMin); // ......... 0
  p_.emplace_back(pRMax); // ......... 1
  p_.emplace_back(pRTor); // ......... 2
  p_.emplace_back(pSPhi); // ......... 3
  p_.emplace_back(pDPhi); // ......... 4
}


void DDI::Torus::stream(std::ostream & os) const
{
  os << " rMin=" << CONVERT_TO( p_[0], cm )
     << " rMax=" << CONVERT_TO( p_[1], deg )
     << " rTor=" << CONVERT_TO( p_[2], deg )
     << " sPhi=" << CONVERT_TO( p_[3], cm )
     << " dPhi=" << CONVERT_TO( p_[4], cm );
}

double DDI::Torus::volume() const
{
  double volume=0;
  
  /* use notation as described in documentation about geant 4 shapes */
  
  // From Geant4: { fCubicVolume = fDPhi*pi*fRtor*(fRmax*fRmax-fRmin*fRmin);
  
  volume = p_[4]*_pi*p_[2]*(p_[1]*p_[1]-p_[0]*p_[0]);
  
  return volume;
}
