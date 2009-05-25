#include "DetectorDescription/Core/src/TruncTubs.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDI::TruncTubs::TruncTubs(double zHalf,
              double rIn, double rOut,
	      double startPhi,
	      double deltaPhi,
	      double cutAtStart,
	      double cutAtDelta,
	      bool cutInside)
  : Solid(ddtrunctubs)
{
  p_.push_back(zHalf);
  p_.push_back(rIn);
  p_.push_back(rOut);
  p_.push_back(startPhi);
  p_.push_back(deltaPhi);  
  p_.push_back(cutAtStart);
  p_.push_back(cutAtDelta);
  p_.push_back(cutInside);
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

