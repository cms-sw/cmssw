#include "DetectorDescription/Core/interface/Tubs.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/Solid.h"

using std::fabs;
using namespace geant_units::operators;

DDI::Tubs::Tubs(double zhalf, double rIn, double rOut, double startPhi, double deltaPhi) : Solid(DDSolidShape::ddtubs) {
  p_.emplace_back(zhalf);
  p_.emplace_back(rIn);
  p_.emplace_back(rOut);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
}

void DDI::Tubs::stream(std::ostream& os) const {
  os << " zhalf=" << convertMmToCm(p_[0]) << " rIn=" << convertMmToCm(p_[1]) << " rOut=" << convertMmToCm(p_[2])
     << " startPhi=" << convertRadToDeg(p_[3]) << " deltaPhi=" << convertRadToDeg(p_[4]);
}

double DDI::Tubs::volume() const {
  double volume = 0;
  double z = 2. * p_[0];
  double rIn = p_[1];
  double rOut = p_[2];
  double phi = p_[4];

  double volume1 = 1_pi * rIn * rIn * z;
  double volume2 = 1_pi * rOut * rOut * z;

  double slice = fabs(phi / (2_pi));

  volume = slice * (volume2 - volume1);

  return volume;
}
