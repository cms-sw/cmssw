#include "DetectorDescription/Core/src/Parallelepiped.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <ostream>
#include <cmath>

void DDI::Parallelepiped::stream(std::ostream & os) const
{
   os << " xhalf[cm]=" << p_[0]/cm
      << " yhalf[cm]=" << p_[1]/cm
      << " zhalf[cm]=" << p_[2]/cm
      << " alpha[deg]=" << p_[3]/deg
      << " theta[deg]=" << p_[4]/deg
      << " phi[deg]=" << p_[5]/deg;

}

double DDI::Parallelepiped::volume() const { 
  double volume = p_[0]*p_[1]*p_[2]*8.0;
  return volume; 
}
