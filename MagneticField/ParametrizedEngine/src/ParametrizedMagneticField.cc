/** \class ParametrizedMagneticField
 *
 *  Magnetic Field based on the Veikko Karimaki's Parametrization
 *
 *  $Date: 2008/02/11 14:20:47 $
 *  $Revision: 1.2 $
 *  \author M. Chiorboli - Universit\`a and INFN Catania
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticField.h"



  

  /// Field value ad specified global point, in Tesla
GlobalVector
ParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  if(gp.perp()>120 || fabs(gp.z()) > 300)
    throw cms::Exception("BadParameters") << "GlobalPoint = " << gp << ", Parametrized Magnetic Field defined only for |z|<300 and r<120" << std::endl;
 
 

  GlobalVector bresult;
  trackerField( gp, a_, l_, bresult);
  return bresult;
}



bool ParametrizedMagneticField::trackerField(const GlobalPoint& gp, double a, double l, GlobalVector& bvec) const {

//
//    B-field in Tracker volume
//    
//     In:   xyz[3]: coordinates (m)
//    Out:  bxyz[3]: Bx,By,Bz    (kG)
//
//    Valid for r<1.2 and |z|<3.25               V.Karimäki 040301
//                                 Updated for CMSSW field 070424
//

// b0=field at centre, l=solenoid length, a=radius (m) (phenomen. parameters) 

// static const float b0=40.681, l=15.284, a=4.6430;   // cmssw
  std::cout << "trackerField inizio a  = " << a << std::endl;
  std::cout << "trackerField inizio l  = " << l << std::endl;
  std::cout << "trackerField inizio gp = " << gp << std::endl;

 static const float b0=40.681;
 static float ap2=4.0*a*a/(l*l);  
 static float hb0=0.5*b0*sqrt(1.0+ap2);
 static float hlova=1.0/sqrt(ap2);
 static float ainv=2.0*hlova/l;
 std::cout << "trackerField b0    = " << b0    << std::endl;
 std::cout << "trackerField ap2   = " << ap2   << std::endl;
 std::cout << "trackerField hb0   = " << hb0   << std::endl;
 std::cout << "trackerField hlova = " << hlova << std::endl;
 std::cout << "trackerField ainv  = " << ainv  << std::endl;

 float xyz[3];//, bxyz[3];
 xyz[0]=0.01*gp.x();
 xyz[1]=0.01*gp.y();
 xyz[2]=0.01*gp.z();

 float r=sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
 float z=xyz[2];
 float az = fabs(z);
 if (r<1.2&&az<3.25) {
  float zainv=z*ainv;
  float rinv=(r>0.0) ? 1.0/r : 0.0;
  float u=hlova-zainv;
  float v=hlova+zainv;
  float fu[5],gv[5];
  ffunkti(u,fu);
  ffunkti(v,gv);
  float rat=r*ainv;
  float corrr=0.00894*r*z*(az-2.2221)*(az-2.2221);
  float corrz=-0.02996*exp(-0.5*(az-1.9820)*(az-1.9820)/(0.78915*0.78915));
  float br=hb0*0.5*rat*(fu[1]-gv[1]-0.125*(fu[3]-gv[3])*rat*rat)+corrr;
  float bz=hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*0.25*rat*rat)+corrz;
  //bxyz[0]=br*xyz[0]*rinv;
  //bxyz[1]=br*xyz[1]*rinv;
  //bxyz[2]=bz;
  ///  GlobalVector bresult(br*xyz[0]*rinv, br*xyz[1]*rinv, bz);
  //  bvec.x()=0.1*br*xyz[0]*rinv;
  //bvec.y()=0.1*br*xyz[1]*rinv;
  //bvec.z()=0.1*bz;
  bvec = 0.1*GlobalVector(br*xyz[0]*rinv, br*xyz[1]*rinv, bz);

  std::cout << "trackerField gp " << gp << std::endl;
  std::cout << "trackerField a "  << a  << std::endl;
  std::cout << "trackerField l "  << l  << std::endl;
  std::cout << "trackerField bvec_x " << bvec.x()  << std::endl;
  std::cout << "trackerField bvec_y " << bvec.y()  << std::endl;
  std::cout << "trackerField bvec_z " << bvec.z()  << std::endl;

  return true;
 }   
 return false;
 // {
 // cout <<"The point is outside the region r<1.2m && |z|<3.0m"<<endl;
 //}
 //return;
}

void ParametrizedMagneticField::ffunkti(float u, float* ff) const {
// Function and its 3 derivatives
 float a,b; 
 a=1.0/(1.0+u*u);
 b=sqrt(a);
 ff[0]=u*b;
 ff[1]=a*b;
 ff[2]=-3.0*u*a*ff[1];
 ff[3]=a*ff[2]*((1.0/u)-4.0*u);
 return;
}

bool
ParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp()<120. && fabs(gp.z())<300.);
}
