#include "DetectorDescription/Core/src/Tubs.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>

using std::fabs;


DDI::Tubs::Tubs(double zhalf,
    	        double rIn, double rOut,	      	      
	        double startPhi, 
	        double deltaPhi)
 : Solid(ddtubs)
{
  p_.push_back(zhalf);
  p_.push_back(rIn);
  p_.push_back(rOut);
  p_.push_back(startPhi);
  p_.push_back(deltaPhi);
}


void DDI::Tubs::stream(std::ostream & os) const
{
  os << " zhalf=" << p_[0]/cm
     << " rIn=" << p_[1]/cm
     << " rOut=" << p_[2]/cm
     << " startPhi=" << p_[3]/deg
     << " deltaPhi=" << p_[4]/deg;		
}

					
double DDI::Tubs::volume() const
{
   double volume=0;
   double z=2.*p_[0];
   double rIn=p_[1];
   double rOut=p_[2];
   double phi=p_[4]/rad;

   double volume1=pi*rIn*rIn*z;
   double volume2=pi*rOut*rOut*z;

   double slice=fabs(phi/(2*pi));

   volume=slice*(volume2-volume1);
  
   return volume;																		
}																 		

