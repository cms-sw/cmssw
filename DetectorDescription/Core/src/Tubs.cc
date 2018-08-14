#include "DetectorDescription/Core/src/Tubs.h"

#include <cmath>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Core/src/Solid.h"

using std::fabs;
using namespace dd::operators;

DDI::Tubs::Tubs(double zhalf,
    	        double rIn, double rOut,	      	      
	        double startPhi, 
	        double deltaPhi)
 : Solid(DDSolidShape::ddtubs)
{
  p_.emplace_back(zhalf);
  p_.emplace_back(rIn);
  p_.emplace_back(rOut);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
}

void DDI::Tubs::stream(std::ostream & os) const
{
  os << " zhalf=" << CONVERT_TO( p_[0], cm )
     << " rIn=" << CONVERT_TO( p_[1], cm )
     << " rOut=" << CONVERT_TO( p_[2], cm )
     << " startPhi=" << CONVERT_TO( p_[3], deg )
     << " deltaPhi=" << CONVERT_TO( p_[4], deg );		
}
					
double DDI::Tubs::volume() const
{
   double volume=0;
   double z=2.*p_[0];
   double rIn=p_[1];
   double rOut=p_[2];
   double phi = CONVERT_TO( p_[4], rad );

   double volume1=1_pi*rIn*rIn*z;
   double volume2=1_pi*rOut*rOut*z;

   double slice=fabs(phi/(2_pi));

   volume=slice*(volume2-volume1);
  
   return volume;																		
}																 		
