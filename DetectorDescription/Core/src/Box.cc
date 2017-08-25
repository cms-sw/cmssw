#include "DetectorDescription/Core/src/Box.h"

#include <ostream>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"

void DDI::Box::stream( std::ostream & os ) const
{
  os << " xhalf[cm]=" << p_[0]/cm
     << " yhalf[cm]=" << p_[1]/cm
     << " zhalf[cm]=" << p_[2]/cm;
}
