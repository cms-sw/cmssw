#include "DetectorDescription/Core/src/Orb.h"

#include <ostream>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"

void DDI::Orb::stream( std::ostream & os ) const
{
  os << " radius[cm]=" << p_[0]/cm;
}
