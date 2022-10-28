#include "DetectorDescription/Core/interface/Trap.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Solid.h"

using std::sqrt;
using namespace geant_units::operators;

DDI::Trap::Trap(double pDz,
                double pTheta,
                double pPhi,
                double pDy1,
                double pDx1,
                double pDx2,
                double pAlp1,
                double pDy2,
                double pDx3,
                double pDx4,
                double pAlp2)
    : Solid(DDSolidShape::ddtrap) {
  p_.emplace_back(pDz);     // ......... 0
  p_.emplace_back(pTheta);  // .. 1
  p_.emplace_back(pPhi);    // ....... 2
  p_.emplace_back(pDy1);    // ........ 3
  p_.emplace_back(pDx1);    // ........ 4
  p_.emplace_back(pDx2);    // ........ 5
  p_.emplace_back(pAlp1);   // ....... 6
  p_.emplace_back(pDy2);    // ........ 7
  p_.emplace_back(pDx3);    // ......... 8
  p_.emplace_back(pDx4);    // ........ 9
  p_.emplace_back(pAlp2);
}

void DDI::Trap::stream(std::ostream& os) const {
  os << " dz=" << convertMmToCm(p_[0]) << " theta=" << convertRadToDeg(p_[1]) << " phi=" << convertRadToDeg(p_[2])
     << " dy1=" << convertMmToCm(p_[3]) << " dx1=" << convertMmToCm(p_[4]) << " dx2=" << convertMmToCm(p_[5])
     << " alpha1=" << convertRadToDeg(p_[6]) << " dy2=" << convertMmToCm(p_[7]) << " dx3=" << convertMmToCm(p_[8])
     << " dx4=" << convertMmToCm(p_[9]) << " alpha2=" << convertRadToDeg(p_[10]);
}

double DDI::Trap::volume() const {
  double volume = 0;

  double dz = p_[0] * 2.;
  double dy1 = p_[3] * 2.;
  double dx1 = p_[4] * 2.;
  double dx2 = p_[5] * 2.;
  double dy2 = p_[7] * 2.;
  double dx3 = p_[8] * 2.;
  double dx4 = p_[9] * 2.;

  volume = ((dx1 + dx2 + dx3 + dx4) * (dy1 + dy2) + (dx4 + dx3 - dx2 - dx1) * (dy2 - dy1) / 3) * dz * 0.125;

  return volume;
}
