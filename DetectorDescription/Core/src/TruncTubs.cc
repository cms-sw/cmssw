#include "DetectorDescription/Core/src/TruncTubs.h"

#include <vector>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

DDI::TruncTubs::TruncTubs(double zHalf,
			  double rIn, double rOut,
			  double startPhi,
			  double deltaPhi,
			  double cutAtStart,
			  double cutAtDelta,
			  bool cutInside)
  : Solid(DDSolidShape::ddtrunctubs)
{
  p_.emplace_back(zHalf);
  p_.emplace_back(rIn);
  p_.emplace_back(rOut);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);  
  p_.emplace_back(cutAtStart);
  p_.emplace_back(cutAtDelta);
  p_.emplace_back(cutInside);
}


void DDI::TruncTubs::stream(std::ostream & os) const
{
  os << " zHalf=" << p_[0]/cm 
     << "cm rIn=" << p_[1]/cm
     << "cm rOut=" << p_[2]/cm
     << "cm startPhi=" << p_[3]/deg
     << "deg deltaPhi=" << p_[4]/deg
     << "deg cutAtStart=" << p_[5]/cm
     << "cm cutAtDelta=" << p_[6]/cm
     << "cm cutInside=" << p_[7];
}

