/** \file
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "OAE85lParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "TkBfield.h"

using namespace std;
using namespace magfieldparam;


OAE85lParametrizedMagneticField::OAE85lParametrizedMagneticField(float b0_, 
								 float a_,
								 float l_) 
  : b0(b0_),
    l(l_),
    a(a_)    
{
  init();
}


OAE85lParametrizedMagneticField::OAE85lParametrizedMagneticField(const edm::ParameterSet& parameters) {
  b0 =  parameters.getParameter<double>("b0");
  l = parameters.getParameter<double>("l");
  a = parameters.getParameter<double>("a");
  init();
}

void OAE85lParametrizedMagneticField::init() {
  ap2=4.0*a*a/(l*l);  
  hb0=0.5*b0*sqrt(1.0+ap2);
  hlova=1.0/sqrt(ap2);
  ainv=2.0*hlova/l;
}



OAE85lParametrizedMagneticField::~OAE85lParametrizedMagneticField() {}


GlobalVector
OAE85lParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  
  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of OAE85lParametrizedMagneticField";
    return GlobalVector();
  }
}


GlobalVector 
OAE85lParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  
// Method formerly named "trackerField"

//
//    B-field in Tracker volume
//    
//     In:   xyz[3]: coordinates (m)
//    Out:  bxyz[3]: Bx,By,Bz    (kG)
//
//    Valid for r<1.2 and |z|<3.0               V.Karimäki 040301
//                                 Updated for CMSSW field 070424
//

// b0=field at centre, l=solenoid length, a=radius (m) (phenomen. parameters) 

  // FIXME: beware of statics...
//  static float ap2=4.0*a*a/(l*l);  
//  static float hb0=0.5*b0*sqrt(1.0+ap2);
//  static float hlova=1.0/sqrt(ap2);
//  static float ainv=2.0*hlova/l;
 float xyz[3];//, bxyz[3];
 // convert to m (CMSSW)
 xyz[0]=0.01*gp.x();
 xyz[1]=0.01*gp.y();
 xyz[2]=0.01*gp.z();

 float r=sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
 float z=xyz[2];
 float az = fabs(z);
 // if (r<1.2&&az<3.0) // NOTE: check omitted, is done already by inTesla (NA)
 {
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
  return  0.1*GlobalVector(br*xyz[0]*rinv, br*xyz[1]*rinv, bz);
 }
 // else {
 // cout <<"The point is outside the region r<1.2m && |z|<3.0m"<<endl;
 //}
 // return GlobalVector();
}

void OAE85lParametrizedMagneticField::ffunkti(float u, float* ff) const {
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
OAE85lParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp()<120. && fabs(gp.z())<300.);
}

