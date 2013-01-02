#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

inline
int diff(float a, float b) {
  approx_math::binary32 ba(a);
  approx_math::binary32 bb(b);
  return ba.i32 - bb.i32;

}


inline
int bits(int a) {
  unsigned int aa = abs(a);
  int b=0; if (a==0) return 0;
  while ( (aa/=2) > 0 )  ++b;
  return (a>0) ? b : -b;

}

void testIt() {
  float y[10];
  int a[10]{0}, h[10]{0}, l[10]{9999999};
  const int N=1000000;
  for (int i=0; i!=N;++i) {
    float x = 1.e-9 + 1.e9*drand48();
    y[0] = logf(x);
    y[2] = unsafe_logf<2>(x);
    y[3] = unsafe_logf<3>(x);
    y[4] = unsafe_logf<4>(x);
    y[5] = unsafe_logf<5>(x);
    y[6] = unsafe_logf<6>(x);
    y[7] = unsafe_logf<7>(x);
    y[8] = unsafe_logf<8>(x);
    for (int k=2; k!=9;k++) {
      a[k]+=diff(y[0],y[k]);
      h[k]=std::max(h[k],diff(y[0],y[k]));
      l[k]=std::min(l[k],diff(y[0],y[k]));
    }
  }
  for (int k=2; k!=9;k++) {
    std::cout << k << ": ave/min/max " << double(a[k])/double(N)  << " " << l[k]  << " " << h[k] << std::endl;
  }
}


inline
float ms(float radLen, float m2, float p2) { 
  constexpr float amscon = 1.8496e-4;    // (13.6MeV)**2
  float e2     = p2 + m2;
  
  float fact = 1.f + 0.038f*log(radLen); fact /= p2; fact *=fact;
  float a = e2*fact;
  return amscon*radLen*a;
}

inline
float msf(float radLen, float m2, float p2) { 
  constexpr float amscon = 1.8496e-4;    // (13.6MeV)**2
  float e2     = p2 + m2;
  
  float fact = 1.f + 0.038f*unsafe_logf<2>(radLen); fact /= p2; fact *=fact;
  float a = e2*fact;
  return amscon*radLen*a;
}


inline
float ms2(float radLen, float m2, float p2) { 
  constexpr float amscon = 1.8496e-4;    // (13.6MeV)**2
  float e2     = p2 + m2;
  float beta2  = p2/e2;
  float fact = 1.f + 0.038f*log(radLen);  fact *=fact;
  float a = fact/(beta2*p2);
  return amscon*radLen*a;
}


template<typename T>
inline 
float bb2(float xi, float m2, float p2) { 

  const T emass = 0.511e-3;
  const T poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const T eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const T delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  T p = sqrt(p2);
  T m = sqrt(m2);
  T e     = sqrt(p2 + m2);
  T beta  = p/e;
  T gamma = e/m;
  T eta2  = beta*gamma; eta2 *= eta2;
  T ratio = emass/m;
  T emax  = 2.*emass*eta2/(1. + 2.*ratio*gamma + ratio*ratio);

  xi /= (beta*beta);

  return xi*(log(2.*emass*emax/(poti*poti)) - 2.*(beta*beta) - delta0);

}

template<typename T>
inline 
float bb(float xi, float m2, float p2) { 

  const T emass = 0.511e-3;
  const T poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const T eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const T delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  T im2 = T(1.)/m2;
  T e2     = p2 + m2;
  T e = sqrt(e2);
  T beta2  = p2/e2;
  T eta2  = p2*im2;
  T ratio2 = (emass*emass)*im2;
  T emax  = T(2.)*emass*eta2/(T(1.) + T(2.)*emass*e*im2 + ratio2);

  xi /= beta2;

  return xi*(log(T(2.)*emass*emax/(poti*poti)) - T(2.)*(beta2) - delta0);

}

template<typename T>
inline 
float bbf(float xi, float m2, float p2) { 

  const T emass = 0.511e-3;
  const T poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const T eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const T delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  T im2 = T(1.)/m2;
  T e2     = p2 + m2;
  T e = sqrt(e2);
  T beta2  = p2/e2;
  T eta2  = p2*im2;
  T ratio2 = (emass*emass)*im2;
  T emax  = T(2.)*emass*eta2/(T(1.) + T(2.)*emass*e*im2 + ratio2);

  xi /= beta2;

  return xi*(unsafe_logf<2>(T(2.)*emass*emax/(poti*poti)) - T(2.)*(beta2) - delta0);

}


template<typename T>
inline 
float bbf2(float xi, float m2, float p2) { 

  const T emass = 0.511e-3;
  const T poti = 16.e-9 * 10.75; // = 16 eV * Z**0.9, for Si Z=14
  const T eplasma = 28.816e-9 * sqrt(2.33*0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const T delta0 = 2*log(eplasma/poti) - 1.;

  // calculate general physics things
  T im2 = T(1.)/m2;
  T e2     = p2; //  + m2;
  T e = sqrt(e2);
  T beta2  =T(1); //  p2/e2;
  T eta2  = p2*im2;
  T ratio2 = (emass*emass)*im2;
  T emax  = T(2.)*emass*eta2/(T(1.) + T(2.)*emass*e*im2 + ratio2);

  xi /= beta2;

  return xi*(unsafe_logf<2>(T(2.)*emass*emax/(poti*poti)) - T(2.)*(beta2) - delta0);

}



template<typename Fun>
void compare(Fun F, Fun F2, Fun Fapx) {
  std::cout << std::endl;

  float m2 = 0.138; m2*=m2;

  int d1=0, d2=0, d3=0;
  int c1=99999999, c2=c1, c3=c1;
  int dm=99999999;
  
  float p2=0.01;
  for (int i=0;i!=6;++i) {
    p2 *=10;
    float rl = 0.001;
    for (int j=0;j!=4;++j) {
      rl *=10;
      float ref = F(rl,m2,p2);
      float rp = F(rl*1.001,m2,p2);
      float rm = F(rl*0.999,m2,p2);
      float apx = Fapx(rl,m2,p2);

      int dd = std::min(abs(diff(rm,ref)),abs(diff(rp,ref)));
      dd -= abs(diff(apx,ref)); // negative if apx-ref is bigger than the uncer-interval
      dm = std::min(dm,dd);

      d1 = std::max(d1,abs(diff(F2(rl,m2,p2),ref)));
      d2 = std::max(d2,abs(diff(apx,ref)));
      d3 = std::max(d3,abs(diff(rp,ref)));
      d3 = std::max(d3,abs(diff(rm,ref)));

      c1 = std::min(c1,abs(diff(F2(rl,m2,p2),ref)));
      c2 = std::min(c2,abs(diff(apx,ref)));
      c3 = std::min(c3,abs(diff(rp,ref)));
      c3 = std::min(c3,abs(diff(rm,ref)));


      // std::cout << diff(ms2(rl,m2,p2),ref) << std::endl;
      // std::cout << diff(msf(rl,m2,p2),ref) << std::endl;
      // std::cout << diff(ms(1.001*rl,m2,p2),ref) << std::endl;
      // std::cout << diff(ms(0.999*rl,m2,p2),ref) << std::endl;
    }
  }

  std::cout  << dm << "," << bits(dm) << std::endl;

  std::cout  << d1 << "," << bits(d1) << " " 
	     << d2 << "," << bits(d2)<< " " 
	     << d3 << "," << bits(d3)<< " " 
	    << std::endl;

  std::cout  << c1 << "," << bits(c1)<< " " 
	     << c2 << "," << bits(c2)<< " " 
	     << c3 << "," << bits(c3)<< " " 
	    << std::endl;

}

int main() {

  std::cout << bits(0) << " "  << bits(1) << " "<< bits(2) << " "<< bits(-31) << " " << bits(32) << std::endl;

  testIt();
  compare(ms,ms2,msf);
  compare(bb<float>,bb2<float>,bbf<float>);
  compare(bb<float>,bb2<float>,bbf2<float>);
  compare(bb<double>,bb2<double>,bbf<double>);

  return 0;
}
