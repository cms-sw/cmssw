#include "DetectorDescription/Core/interface/Sphere.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <cmath>
#include <ostream>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Solid.h"

using namespace geant_units;
using namespace geant_units::operators;

DDI::Sphere::Sphere(
    double innerRadius, double outerRadius, double startPhi, double deltaPhi, double startTheta, double deltaTheta)
    : Solid(DDSolidShape::ddsphere) {
  p_.emplace_back(innerRadius);
  p_.emplace_back(outerRadius);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  p_.emplace_back(startTheta);
  p_.emplace_back(deltaTheta);
}

void DDI::Sphere::stream(std::ostream& os) const {
  os << " innerRadius=" << convertMmToCm(p_[0]) << " outerRadius=" << convertMmToCm(p_[1])
     << " startPhi=" << convertRadToDeg(p_[2]) << " deltaPhi=" << convertRadToDeg(p_[3])
     << " startTheta=" << convertRadToDeg(p_[4]) << " deltaTheta=" << convertRadToDeg(p_[5]);
}

double DDI::Sphere::volume() const {
  double volume(0.);
  if (std::fabs(p_[3]) <= 2._pi && std::fabs(p_[5]) <= piRadians) {
    volume = std::fabs((p_[1] * p_[1] * p_[1] - p_[0] * p_[0] * p_[0]) / 3. *
                       (std::cos(p_[4] + p_[5]) - std::cos(p_[4])) * p_[3]);
  } else if (std::fabs(p_[3]) <= 2._pi && std::fabs(p_[5]) > piRadians) {
    volume = std::fabs((p_[1] * p_[1] * p_[1] - p_[0] * p_[0] * p_[0]) / 3. *
                       (std::cos(p_[4] + p_[5] - 180._deg) - std::cos(p_[4])) * p_[3]);
  } else if (std::fabs(p_[3]) > 2._pi && std::fabs(p_[5]) <= piRadians) {
    volume = std::fabs((p_[1] * p_[1] * p_[1] - p_[0] * p_[0] * p_[0]) / 3. *
                       (std::cos(p_[4] + p_[5]) - std::cos(p_[4])) * (p_[3] - p_[2]));
  }
  return volume;
}
