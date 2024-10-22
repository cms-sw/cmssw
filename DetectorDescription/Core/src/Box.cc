#include "DetectorDescription/Core/interface/Box.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <ostream>

using namespace geant_units::operators;

void DDI::Box::stream(std::ostream& os) const {
  os << " xhalf[cm]=" << convertMmToCm(p_[0]) << " yhalf[cm]=" << convertMmToCm(p_[1])
     << " zhalf[cm]=" << convertMmToCm(p_[2]);
}
