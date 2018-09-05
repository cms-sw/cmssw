#include "DetectorDescription/Core/src/Cons.h"
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <cmath>
#include <ostream>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"

using namespace dd;
using namespace dd::operators;

DDI::Cons::Cons(double zhalf,
		double rInMinusZ,
		double rOutMinusZ,
		double rInPlusZ,
		double rOutPlusZ,
		double startPhi,
		double deltaPhi)
  : Solid(DDSolidShape::ddcons)
{
  p_.emplace_back(zhalf);
  p_.emplace_back(rInMinusZ);
  p_.emplace_back(rOutMinusZ);
  p_.emplace_back(rInPlusZ);
  p_.emplace_back(rOutPlusZ);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
}

void
DDI::Cons::stream( std::ostream & os ) const
{
  os << " zhalf=" << CONVERT_TO( p_[0], cm )
     << " rIn-Z=" << CONVERT_TO( p_[1], cm )
     << " rOut-Z=" << CONVERT_TO( p_[2], cm )
     << " rIn+Z=" << CONVERT_TO( p_[3], cm )
     << " rOut+Z=" << CONVERT_TO( p_[4], cm )
     << " startPhi=" << CONVERT_TO( p_[5], deg )
     << " deltaPhi=" << CONVERT_TO( p_[6], deg );
}

double DDI::Cons::volume() const
{
  /* zhalf is the half length of the cone,
     phiTo is always clockwise rotated from phiFrom 
     rInMinusZ is always smaller than rOutMinusZ (same for r*PlusZ)
     They are the distances relative to the rotation axes */

  /* calculation normalize from 0 to z */

  /* The function f=rInMinusZ+((rInPlusZ-rInMinusZ)/z)*x defines
     the radius of the the rotation from 0 to z. Raised to the power
     of 2 integrated on x from 0 to z. Multiplied by pi, gives the
     volume that needs to substracted from the other volume */ 
     
  /* f^2=rInMinusZ*rInMinusZ+2*rInMinusZ*((rInPlusZ-rInMinusZ)/z)*x+((rInPlusZ-rInMinusZ)*(rInPlusZ-rInMinusZ)*x*x)/(z*z) */

  /* primitive of f^2 is: rInMinusZ*rInMinusZ*x+rInMinusZ*((rInPlusZ-rInMinusZ)/z)*(x*x)+(rInPlusZ-rInMinusZ)*(rInPlusZ-rInMinusZ)*(x*x*x)/(3*z*z) */

  /*integration from 0 to z yields: pi*( rInMinusZ*rInMinusZ*z+rInMinusZ*(rInPlusZ-rInMinusZ)*z+((rInPlusZ-rInMinusZ)*(rInPlusZ-rInMinusZ)*z)/(3) ) */

   double zhalf=p_[0]; 
   double rInMinusZ=p_[1]; 
   double rOutMinusZ=p_[2]; 
   double rInPlusZ=p_[3]; 
   double rOutPlusZ=p_[4];
   //double phiFrom=p_[5]/rad;
   double deltaPhi=std::abs( CONVERT_TO( p_[6], rad )); 
   double z=2*zhalf;

  double volume1=1_pi*(rInPlusZ*rInPlusZ+rInMinusZ*rInMinusZ+rInMinusZ*rInPlusZ)*z/3;

  double volume2=1_pi*(rOutPlusZ*rOutPlusZ+rOutMinusZ*rOutMinusZ+rOutMinusZ*rOutPlusZ)*z/3;
  
  double slice=deltaPhi/(2_pi);
  double volume=slice*(volume2-volume1);

  return volume;

}


