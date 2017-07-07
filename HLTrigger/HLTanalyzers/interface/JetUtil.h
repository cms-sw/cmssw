#ifndef JET_JETUTIL_H
#define JET_JETUTIL_H 1

#include <cmath>

class PtGreater {
  public:
  template <typename T> bool operator () (const T& i, const T& j) {
    return (i.pt() > j.pt());
  }
};

class EtGreater {
  public:
  template <typename T> bool operator () (const T& i, const T& j) {
    return (i.et() > j.et());
  }
};

class JetUtil{
 public:

static double Phi_0_2pi(double x) {
  while (x >= 2*M_PI) x -= 2*M_PI;
  while (x <     0.)  x += 2*M_PI;
  return x;
}

static double Phi_mpi_pi(double x) {
   while (x >= M_PI) x -= 2*M_PI;
   while (x < -M_PI) x += 2*M_PI;
   return x;
}

static double dPhi(double phi1,double phi2){
   phi1=Phi_0_2pi(phi1);
   phi2=Phi_0_2pi(phi2);
   return Phi_mpi_pi(phi1-phi2);
}

static double radius(double eta1, double phi1,double eta2, double phi2){
 
  const double TWOPI= 2.0*M_PI;
 
  phi1=Phi_0_2pi(phi1);
  phi2=Phi_0_2pi(phi2);
 
  double dphi=Phi_0_2pi(phi1-phi2);
  dphi = TMath::Min(dphi,TWOPI-dphi);
  double deta = eta1-eta2;
 
  return sqrt(deta*deta+dphi*dphi);
}

template <typename T1,typename T2> static double radius(const T1&  t1,const T2& t2){
  return radius(t1->eta(),t1->phi(),t2->eta(),t2->phi());
}
};
#endif
