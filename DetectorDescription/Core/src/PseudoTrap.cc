#include "DetectorDescription/Core/src/PseudoTrap.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"



void DDI::PseudoTrap::stream(std::ostream & os) const
{
  os << " x1[cm]=" << p_[0]/cm
     << " x2[cm]=" << p_[1]/cm
     << " y1[cm]=" << p_[2]/cm
     << " y2[cm]=" << p_[3]/cm
     << " z[cm]=" << p_[4]/cm
     << " radius[cm]=" << p_[5]/cm;
     
  if (p_[6])
     os << " minusZ=[yes]";
  else
     os << " minusZ=[no]";
}
