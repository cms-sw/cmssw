#include "DetectorDescription/Core/src/Trap.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <cmath>

using std::sqrt;


DDI::Trap::Trap( double pDz, 
                 double pTheta,
                 double pPhi,
                 double pDy1, double pDx1,double pDx2,
                 double pAlp1,
                 double pDy2, double pDx3, double pDx4,
                 double pAlp2 )
 : Solid(ddtrap) 
{		 
  p_.push_back(pDz); // ......... 0
  p_.push_back(pTheta); // .. 1
  p_.push_back(pPhi); // ....... 2
  p_.push_back(pDy1); // ........ 3
  p_.push_back(pDx1); // ........ 4
  p_.push_back(pDx2); // ........ 5
  p_.push_back(pAlp1); // ....... 6
  p_.push_back(pDy2); // ........ 7
  p_.push_back(pDx3); // ......... 8
  p_.push_back(pDx4); // ........ 9
  p_.push_back(pAlp2);
}


void DDI::Trap::stream(std::ostream & os) const
{
  os << " dz=" << p_[0]/cm
     << " theta=" << p_[1]/deg
     << " phi=" << p_[2]/deg
     << " dy1=" << p_[3]/cm
     << " dx1=" << p_[4]/cm
     << " dx2=" << p_[5]/cm
     << " alpha1=" << p_[6]/deg
     << " dy2=" << p_[7]/cm
     << " dx3=" << p_[8]/cm
     << " dx4=" << p_[9]/cm
     << " alpha2=" << p_[10]/deg;
}

double DDI::Trap::volume() const
{
 double volume=0;

  /* use notation as described in documentation about geant 3 shapes */
  /* we do not need all the parameters.*/

  double Dz=p_[0];
  double H1=p_[3];
  double Bl1=p_[4];
  double Tl1=p_[5];
  double H2=p_[7];
  double Bl2=p_[8];
  double Tl2=p_[9];

  double z=2*Dz;

  /* the area of a trapezoid with one side of length 2*Bl1 and other side 2*Tl1,     and height 2*H1 is 0.5*(2*Bl1+2*Tl1)*2H1=2*H1(Bl1+Tl1) */

  /* the volume of a geometry defined by 2 2D parallel trapezoids is (in this case the integral over the area of a trapezoid that is defined as function x between these two trapezoids */

  /* the following formula describes this parmeterized area in x. x=0: trapezoid defined by H1,Bl1,Tl1,  x=z: trapezoid defined by H2,Bl2,Tl2 */

  /* area(x)=2*(H1+x/z*(H2-H1))*(Bl1+x/z*(Bl2-Bl1)+Tl1+x/z*(Tl2-Tl1)) */
 
 /* primitive of area(x):
    (2*H1*Bl1+2*H1*Tl1)*x+(H1*Bl2-2*H1*Bl1+H1*Tl2-2*H1*Tl1+H2*Bl1+H2*Tl1+H2*Tl2-H2*Tl1)*x*x/z+(2/3)*(H2*Bl2-H2*Bl1-H1*Bl2+H1*Bl1-H1*Tl2+H1*Tl1)*x*x*x/(z*z)   */

// volume=(2*H1*Bl1+2*H1*Tl1)*z+(H1*Bl2-2*H1*Bl1+H1*Tl2-2*H1*Tl1+H2*Bl1+H2*Tl1+H2*Tl2-H2*Tl1)*z*z+(2/3)*(H2*Bl2-H2*Bl1-H1*Bl2+H1*Bl1-H1*Tl2+H1*Tl1)*z*z*z; 
  volume=(2*H1*Bl1+2*H1*Tl1)*z+(H1*Bl2-2*H1*Bl1+H1*Tl2-2*H1*Tl1+H2*Bl1+H2*Tl1+H2*Tl2-H2*Tl1)*z+(2/3)*(H2*Bl2-H2*Bl1-H1*Bl2+H1*Bl1-H1*Tl2+H1*Tl1)*z; 


 /* 
    Alternative:
    A ... height of bottom trapez, B ... middle line perpendicular to A
    a ... height of top trapez,    b ... middle line perpendicular to a
    H ... heigt of the solid
    
    V = H/3. * ( A*B + 0.5 * ( A*b + B*a ) + a*b ) <-- this is wrong ..
    V = H/3 * ( A*B + sqrt( A*B*a*b ) + a*b )
 */ 
//   double A = 2.*p_[3];
//   double B = p_[4] + p_[5];
//   double a = 2.*p_[7];
//   double b = p_[8] + p_[9];
  

//   double volu_alt = 2.*p_[0]/3. * ( A*B + sqrt ( A*b*B*a ) + a*b );
//   DCOUT('V', "alternative-volume=" << volu_alt/m3 << std::endl);
  
  //DCOUT_V('C',"DC: solid=" << this->ddname() << " vol=" << volume << " vol_a=" << volu_alt << " d=" << (volume-volu_alt)/volume*100. << "%");
  return volume;
}
