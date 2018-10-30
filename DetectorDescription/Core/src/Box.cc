#include "DetectorDescription/Core/src/Box.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <ostream>

using namespace dd::operators;

void DDI::Box::stream( std::ostream & os ) const
{
  os << " xhalf[cm]=" << CONVERT_TO( p_[0], cm )
     << " yhalf[cm]=" << CONVERT_TO( p_[1], cm )
     << " zhalf[cm]=" << CONVERT_TO( p_[2], cm );
}
