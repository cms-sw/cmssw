#include <MagneticField/ParametrizedEngine/src/TkBfield.h>

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;
using namespace magfieldparam;


TkBfield::TkBfield(string fld) {
//  double p1[]={4.79358,17.7694,2.02881,0.0208176,0.000284306,2.47374,0.0085307,2.30347,4.03922};  // 2.0T
//  double p2[]={4.36256,15.7269,3.03226,0.0195461,0.000468156,2.51173,0.0118981,2.31428,3.49307};  // 3.0T
//  double p3[]={4.25013,15.2184,3.52570,0.0180985,0.000553656,2.52681,0.0135540,2.31768,3.35313};  // 3.5T
//  double p4[]={4.19476,14.9829,3.82161,0.0176056,0.000600249,2.53309,0.0144965,2.31895,3.29954};  // 3.8T
//  double p5[]={4.15918,14.8424,4.02377,0.0173738,0.000631032,2.53689,0.0151167,2.31917,3.26681};  // 4.0T
  double p1[]={4.78600,17.7443,2.02356,0.0207752,0.000292842,2.45264,0.00334335,2.16768,1.80184}; // 2.0T-2G
  double p2[]={4.35122,15.7041,3.02623,0.0195655,0.000478392,2.50056,0.00598570,2.18720,1.82336}; // 3.0T-2G
  double p3[]={4.23028,15.1896,3.51933,0.0181075,0.000560562,2.52665,0.00735058,2.20449,1.84352}; // 3.5T-2G
  double p4[]={4.17523,14.9553,3.81501,0.0176136,0.000606742,2.53488,0.00807716,2.20819,1.84780}; // 3.8T-2G
  double p5[]={4.13973,14.8154,4.01702,0.0173780,0.000637390,2.53979,0.00856027,2.21038,1.85026}; // 4.0T-2G
  prm[0]=0;
  if (fld=="2_0T") for (int i=0; i<9; i++) prm[i]=p1[i];
  if (fld=="3_0T") for (int i=0; i<9; i++) prm[i]=p2[i];
  if (fld=="3_5T") for (int i=0; i<9; i++) prm[i]=p3[i];
  if (fld=="3_8T") for (int i=0; i<9; i++) prm[i]=p4[i];
  if (fld=="4_0T") for (int i=0; i<9; i++) prm[i]=p5[i];
  //  cout<<std::endl<<"Instantiation of TkBfield with key "<<fld<<endl;
  if (!prm[0]) {
    throw cms::Exception("BadParameters") << "Undefined key - " // abort!"<<endl;
    <<"Defined keys are: \"2_0T\" \"3_0T\" \"3_5T\" \"3_8T\" and \"4_0T\""<<endl;
    // exit(1);
  }
  ap2=4*prm[0]*prm[0]/(prm[1]*prm[1]);  
  hb0=0.5*prm[2]*sqrt(1.0+ap2);
  hlova=1/sqrt(ap2);
  ainv=2*hlova/prm[1];
//  coeff=1/(prm[8]*prm[8]);
}

void TkBfield::Bcyl (const double* x) {
  double r=sqrt(x[0]*x[0]+x[1]*x[1]);
  double z=x[2];
  if (r<1.1&&fabs(z)<3) {
    z-=prm[3];                    // max Bz point is shifted in z
    double az=fabs(z);
    double zainv=z*ainv;
    double u=hlova-zainv;
    double v=hlova+zainv;
    double fu[4],gv[4];
    ffunkti(u,fu);
    ffunkti(v,gv);
    double rat=r*ainv;
    double rat2=rat*rat;
    Bw[0]=hb0*rat*(fu[1]-gv[1]-(fu[3]-gv[3])*rat2/8)/2;
    Bw[1]=0;
    Bw[2]=hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*rat2/4);
    double corBr= prm[4]*r*z*(az-prm[5])*(az-prm[5]);
//    double corBz=-prm[6]*exp(coeff*(az-prm[7])*(az-prm[7]));
//    double corBz=-prm[6]/(1+coeff*(az-prm[7])*(az-prm[7]));
    double corBz=-prm[6]*(exp(-(z-prm[7])*(z-prm[7])/(prm[8]*prm[8]))
                        + exp(-(z+prm[7])*(z+prm[7])/(prm[8]*prm[8]))); // double Gaussian
    Bw[0]+=corBr;
    Bw[2]+=corBz;
  } else {
    cout <<"TkBfield: The point is outside the region r<1.1m && |z|<3.0m"<<endl;
  }
}

void TkBfield::ffunkti(const double u, double* ff) {
// Function and its 3 derivatives
  double a,b,a2,u2;
  u2=u*u; 
  a=1/(1+u2);
  a2=-3*a*a;
  b=sqrt(a);
  ff[0]=u*b;
  ff[1]=a*b;
  ff[2]=a2*ff[0];
  ff[3]=a2*ff[1]*(1-4*u2);
}

void TkBfield::getBrfz(const double *x, double *Brfz) {
  Bcyl(x);
  for (int i=0; i<3; i++) Brfz[i]=Bw[i];
}

void TkBfield::getBxyz(const double *x, double *Bxyz) {
  Bcyl(x);
  double r=sqrt(x[0]*x[0]+x[1]*x[1]);
  double rinv=(r>0) ? 1/r:0;
  Bxyz[0]=Bw[0]*x[0]*rinv;
  Bxyz[1]=Bw[0]*x[1]*rinv;
  Bxyz[2]=Bw[2];
}
