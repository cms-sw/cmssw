#include "DetectorDescription/Core/src/PseudoTrap.h"
#include "DataFormats/Math/interface/Units.h"

using namespace cms_units::operators;

void DDI::PseudoTrap::stream(std::ostream & os) const
{
  os << " x1[cm]=" << CMS_CONVERT_TO( p_[0], cm )
     << " x2[cm]=" << CMS_CONVERT_TO( p_[1], cm )
     << " y1[cm]=" << CMS_CONVERT_TO( p_[2], cm )
     << " y2[cm]=" << CMS_CONVERT_TO( p_[3], cm )
     << " z[cm]=" << CMS_CONVERT_TO( p_[4], cm )
     << " radius[cm]=" << CMS_CONVERT_TO( p_[5], cm );
     
  if (p_[6])
     os << " minusZ=[yes]";
  else
     os << " minusZ=[no]";
}
