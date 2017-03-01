#include "DetectorDescription/Core/src/CutTubs.h"

#include <cmath>
#include <vector>

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

DDI::CutTubs::CutTubs(double zhalf,
		      double rIn, double rOut,	      	      
		      double startPhi, 
		      double deltaPhi,
		      double lx, double ly, double lz,
		      double tx, double ty, double tz)
  : Solid(ddcuttubs)
{
  p_.push_back(zhalf);
  p_.push_back(rIn);
  p_.push_back(rOut);
  p_.push_back(startPhi);
  p_.push_back(deltaPhi);
  p_.push_back(lx);
  p_.push_back(ly);
  p_.push_back(lz);
  p_.push_back(tx);
  p_.push_back(ty);
  p_.push_back(tz);
}

void DDI::CutTubs::stream(std::ostream & os) const
{
  os << " zhalf=" << p_[0]/cm
     << " rIn=" << p_[1]/cm
     << " rOut=" << p_[2]/cm
     << " startPhi=" << p_[3]/deg
     << " deltaPhi=" << p_[4]/deg
     << " Outside Normal at -z (" << p_[5] << "," << p_[6] << "," << p_[7] << ")"
     << " Outside Normal at +z (" << p_[8] << "," << p_[9] << "," << p_[10] << ")";		
}
