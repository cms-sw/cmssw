#include "DetectorDescription/Core/interface/TruncTubs.h"

#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/Solid.h"

using namespace geant_units::operators;

DDI::TruncTubs::TruncTubs(double zHalf,
                          double rIn,
                          double rOut,
                          double startPhi,
                          double deltaPhi,
                          double cutAtStart,
                          double cutAtDelta,
                          bool cutInside)
    : Solid(DDSolidShape::ddtrunctubs) {
  p_.emplace_back(zHalf);
  p_.emplace_back(rIn);
  p_.emplace_back(rOut);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  p_.emplace_back(cutAtStart);
  p_.emplace_back(cutAtDelta);
  p_.emplace_back(cutInside);
}

void DDI::TruncTubs::stream(std::ostream& os) const {
  os << " zHalf=" << convertMmToCm(p_[0]) << "cm rIn=" << convertMmToCm(p_[1]) << "cm rOut=" << convertMmToCm(p_[2])
     << "cm startPhi=" << convertRadToDeg(p_[3]) << "deg deltaPhi=" << convertRadToDeg(p_[4])
     << "deg cutAtStart=" << convertMmToCm(p_[5]) << "cm cutAtDelta=" << convertMmToCm(p_[6])
     << "cm cutInside=" << p_[7];
}
