#include "DetectorDescription/Core/src/Torus.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <cmath>

using std::sqrt;


DDI::Torus::Torus( double pRMin,
		   double pRMax,
		   double pRTor,
		   double pSPhi,
		   double pDPhi
		   )
 : Solid(ddtorus) 
{		 
  p_.push_back(pRMin); // ......... 0
  p_.push_back(pRMax); // ......... 1
  p_.push_back(pRTor); // ......... 2
  p_.push_back(pSPhi); // ......... 3
  p_.push_back(pDPhi); // ......... 4
}


void DDI::Torus::stream(std::ostream & os) const
{
  os << " rMin=" << p_[0]/cm
     << " rMax=" << p_[1]/deg
     << " rTor=" << p_[2]/deg
     << " sPhi=" << p_[3]/cm
     << " dPhi=" << p_[4]/cm;
}

double DDI::Torus::volume() const
{
  double volume=0;
  
  /* use notation as described in documentation about geant 4 shapes */
  
  // From Geant4: { fCubicVolume = fDPhi*pi*fRtor*(fRmax*fRmax-fRmin*fRmin);
  
  volume = p_[4]*pi*p_[2]*(p_[1]*p_[1]-p_[0]*p_[0]);
  
  return volume;
}
