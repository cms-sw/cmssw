#include "DetectorDescription/Core/interface/Torus.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Solid.h"

using std::sqrt;
using namespace geant_units;
using namespace geant_units::operators;

DDI::Torus::Torus(double pRMin, double pRMax, double pRTor, double pSPhi, double pDPhi) : Solid(DDSolidShape::ddtorus) {
  p_.emplace_back(pRMin);  // ......... 0
  p_.emplace_back(pRMax);  // ......... 1
  p_.emplace_back(pRTor);  // ......... 2
  p_.emplace_back(pSPhi);  // ......... 3
  p_.emplace_back(pDPhi);  // ......... 4
}

void DDI::Torus::stream(std::ostream& os) const {
  os << " rMin=" << convertMmToCm(p_[0]) << " rMax=" << convertRadToDeg(p_[1]) << " rTor=" << convertRadToDeg(p_[2])
     << " sPhi=" << convertMmToCm(p_[3]) << " dPhi=" << convertMmToCm(p_[4]);
}

double DDI::Torus::volume() const {
  double volume = 0;

  /* use notation as described in documentation about geant 4 shapes */

  // From Geant4: { fCubicVolume = fDPhi*pi*fRtor*(fRmax*fRmax-fRmin*fRmin);

  volume = p_[4] * piRadians * p_[2] * (p_[1] * p_[1] - p_[0] * p_[0]);

  return volume;
}
