#include "DetectorDescription/Core/interface/PseudoTrap.h"
#include "DataFormats/Math/interface/GeantUnits.h"

using namespace geant_units::operators;

void DDI::PseudoTrap::stream(std::ostream& os) const {
  os << " x1[cm]=" << convertMmToCm(p_[0]) << " x2[cm]=" << convertMmToCm(p_[1]) << " y1[cm]=" << convertMmToCm(p_[2])
     << " y2[cm]=" << convertMmToCm(p_[3]) << " z[cm]=" << convertMmToCm(p_[4])
     << " radius[cm]=" << convertMmToCm(p_[5]);

  if (p_[6])
    os << " minusZ=[yes]";
  else
    os << " minusZ=[no]";
}
