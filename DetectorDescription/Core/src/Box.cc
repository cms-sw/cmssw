#include "DetectorDescription/Core/src/Box.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <ostream>

using namespace geant_units::operators;

void DDI::Box::stream( std::ostream & os ) const
{
  os << " xhalf[cm]=" << CMS_CONVERT_TO( p_[0], cm )
     << " yhalf[cm]=" << CMS_CONVERT_TO( p_[1], cm )
     << " zhalf[cm]=" << CMS_CONVERT_TO( p_[2], cm );
}
