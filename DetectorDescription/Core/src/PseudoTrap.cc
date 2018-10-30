#include "DetectorDescription/Core/src/PseudoTrap.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

using namespace dd::operators;

void DDI::PseudoTrap::stream(std::ostream & os) const
{
  os << " x1[cm]=" << CONVERT_TO( p_[0], cm )
     << " x2[cm]=" << CONVERT_TO( p_[1], cm )
     << " y1[cm]=" << CONVERT_TO( p_[2], cm )
     << " y2[cm]=" << CONVERT_TO( p_[3], cm )
     << " z[cm]=" << CONVERT_TO( p_[4], cm )
     << " radius[cm]=" << CONVERT_TO( p_[5], cm );
     
  if (p_[6])
     os << " minusZ=[yes]";
  else
     os << " minusZ=[no]";
}
