#include "TkBfield.h"

#include "FWCore/Utilities/interface/Exception.h"

// #include <iostream>
// #include <iomanip>
#include <cmath>
#include "DataFormats/Math/interface/SSEVec.h"

using namespace magfieldparam;


TkBfield::TkBfield(std::string fld) {
  double p1[]={4.90541,17.8768,2.02355,0.0210538,0.000321885,2.37511,0.00326725,2.07656,1.71879}; // 2.0T-2G
  double p2[]={4.41982,15.7732,3.02621,0.0197814,0.000515759,2.43385,0.00584258,2.11333,1.76079}; // 3.0T-2G
  double p3[]={4.30161,15.2586,3.51926,0.0183494,0.000606773,2.45110,0.00709986,2.12161,1.77038}; // 3.5T-2G
  double p4[]={4.24326,15.0201,3.81492,0.0178712,0.000656527,2.45818,0.00778695,2.12500,1.77436}; // 3.8T-2G
  double p5[]={4.21136,14.8824,4.01683,0.0175932,0.000695541,2.45311,0.00813447,2.11688,1.76076}; // 4.0T-2G
  prm[0]=0;
  if (fld=="2_0T") for (int i=0; i<9; i++) prm[i]=p1[i];
  if (fld=="3_0T") for (int i=0; i<9; i++) prm[i]=p2[i];
  if (fld=="3_5T") for (int i=0; i<9; i++) prm[i]=p3[i];
  if (fld=="3_8T") for (int i=0; i<9; i++) prm[i]=p4[i];
  if (fld=="4_0T") for (int i=0; i<9; i++) prm[i]=p5[i];
  //  cout<<std::endl<<"Instantiation of TkBfield with key "<<fld<<endl;
  if (!prm[0]) {
    throw cms::Exception("BadParameters") << "Undefined key - " // abort!\n";
    <<"Defined keys are: \"2_0T\" \"3_0T\" \"3_5T\" \"3_8T\" and \"4_0T\"\n";
    // exit(1);
  }
  ap2=4*prm[0]*prm[0]/(prm[1]*prm[1]);  
  hb0=0.5*prm[2]*std::sqrt(1.0+ap2);
  hlova=1/std::sqrt(ap2);
  ainv=2*hlova/prm[1];
  coeff=1/(prm[8]*prm[8]);
}

namespace {

  template<typename T>
  inline void ffunkti(T u, T * __restrict__ ff) __attribute__((always_inline)) __attribute__ ((pure));
 
  template<typename T>
  inline void ffunkti(T u, T * __restrict__ ff) {
    // Function and its 3 derivatives
    T a,b,a2,u2;
    u2=u*u; 
    a=T(1)/(T(1)+u2);
    a2=-3*a*a;
    b=mathSSE::sqrt(a);
    ff[0]=u*b;
    ff[1]=a*b;
    ff[2]=a2*ff[0];
    ff[3]=a2*ff[1]*(T(1)-4*u2);
  }
}

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) || defined(__clang__)

void  TkBfield::Bcyl(double r, double z, double * __restrict__ Bw)  const{
  z-=prm[3];                    // max Bz point is shifted in z
  double az=std::abs(z);
  double zainv=z*ainv;

  mathSSE::Vec2D uv(hlova-zainv,hlova+zainv);
  mathSSE::Vec2D fuv[4];
  ffunkti(uv,fuv);

  double rat=0.5*r*ainv;
  double rat2=rat*rat;
  Bw[0]=hb0*rat*(fuv[1][0]-fuv[1][1]-(fuv[3][0]-fuv[3][1])*rat2*0.5);
  Bw[1]=0;
  Bw[2]=hb0*(fuv[0][0]+fuv[0][1]-(fuv[2][0]+fuv[2][1])*rat2);

  double corBr= prm[4]*r*z*(az-prm[5])*(az-prm[5]);
  double corBz=-prm[6]*(exp(-(z-prm[7])*(z-prm[7])*coeff) +
			exp(-(z+prm[7])*(z+prm[7])*coeff)
			); // double Gaussian
  Bw[0]+=corBr;
  Bw[2]+=corBz;
}

#else

void  TkBfield::Bcyl(double r, double z, double * __restrict__ Bw)  const{
  //  if (r<1.15&&fabs(z)<2.8) // NOTE: check omitted, is done already by the wrapper! (NA)
  z-=prm[3];                    // max Bz point is shifted in z
  double az=std::abs(z);
  double zainv=z*ainv;
  double u=hlova-zainv;
  double v=hlova+zainv;
  double fu[4],gv[4];
  ffunkti(u,fu);
  ffunkti(v,gv);
  double rat=0.5*r*ainv;
  double rat2=rat*rat;
  Bw[0]=hb0*rat*(fu[1]-gv[1]-(fu[3]-gv[3])*rat2*0.5);
  Bw[1]=0;
  Bw[2]=hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*rat2);
  double corBr= prm[4]*r*z*(az-prm[5])*(az-prm[5]);
  //    double corBz=-prm[6]*exp(coeff*(az-prm[7])*(az-prm[7]));
  //    double corBz=-prm[6]/(1+coeff*(az-prm[7])*(az-prm[7]));
  //  double corBz=-prm[6]*(exp(-(z-prm[7])*(z-prm[7])/(prm[8]*prm[8]))
  //			+ exp(-(z+prm[7])*(z+prm[7])/(prm[8]*prm[8]))); // double Gaussian
  double corBz=-prm[6]*(exp(-(z-prm[7])*(z-prm[7])*coeff) +
			exp(-(z+prm[7])*(z+prm[7])*coeff)
			); // double Gaussian
  Bw[0]+=corBr;
  Bw[2]+=corBz;
  //   } else {
  //     cout <<"TkBfield: The point is outside the region r<1.15m && |z|<2.8m"<<endl;
}
#endif


void TkBfield::getBrfz(double const  * __restrict__ x, double * __restrict__ Brfz)  const {
  Bcyl(sqrt(x[0]*x[0]+x[1]*x[1]),x[2], Brfz);
}

void TkBfield::getBxyz(double const  * __restrict__ x, double * __restrict__ Bxyz)  const {
  double Bw[3];
  double r=sqrt(x[0]*x[0]+x[1]*x[1]);
  Bcyl(r, x[2], Bw);
  double rinv=(r>0) ? 1/r:0;
  Bxyz[0]=Bw[0]*x[0]*rinv;
  Bxyz[1]=Bw[0]*x[1]*rinv;
  Bxyz[2]=Bw[2];
}

