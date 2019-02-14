#include "DetectorDescription/Core/src/CutTubs.h"
#include "DataFormats/Math/interface/Units.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

using namespace cms_units;
using namespace cms_units::operators;

DDI::CutTubs::CutTubs( double zhalf,
		       double rIn, double rOut,	      	      
		       double startPhi, 
		       double deltaPhi,
		       double lx, double ly, double lz,
		       double tx, double ty, double tz )
  : Solid(DDSolidShape::ddcuttubs)
{
  p_.emplace_back(zhalf);
  p_.emplace_back(rIn);
  p_.emplace_back(rOut);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  p_.emplace_back(lx);
  p_.emplace_back(ly);
  p_.emplace_back(lz);
  p_.emplace_back(tx);
  p_.emplace_back(ty);
  p_.emplace_back(tz);
}

void DDI::CutTubs::stream(std::ostream & os) const
{
  os << " zhalf=" << CMS_CONVERT_TO( p_[0], cm )
     << " rIn=" << CMS_CONVERT_TO( p_[1], cm )
     << " rOut=" << CMS_CONVERT_TO( p_[2], cm )
     << " startPhi=" << CMS_CONVERT_TO( p_[3], deg )
     << " deltaPhi=" << CMS_CONVERT_TO( p_[4], deg )
     << " Outside Normal at -z (" << p_[5] << "," << p_[6] << "," << p_[7] << ")"
     << " Outside Normal at +z (" << p_[8] << "," << p_[9] << "," << p_[10] << ")";		
}
