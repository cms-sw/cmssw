#pragma GCC diagnostic ignored "-Wformat"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace almostEqualDetail {
  union fasi {
    int i;
    float f;
  };
  
  union dasi {
    long long i;
    double f;
  };
  
}

inline int intDiff(float a, float b)
{
  using namespace almostEqualDetail;
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.
  fasi fa; fa.f = a;
  // Make aInt lexicographically ordered as a twos-complement int
  if (fa.i < 0) fa.i = 0x80000000 - fa.i;
  // Make bInt lexicographically ordered as a twos-complement int
  fasi fb; fb.f = b;
  if (fb.i < 0) fb.i = 0x80000000 - fb.i;
  return std::abs(fa.i - fb.i);
}


inline long long intDiff(double a, double b)
{
  using namespace almostEqualDetail;
  dasi fa; fa.f = a;
  // Make aInt lexicographically ordered as a twos-complement int
  if (fa.i < 0) fa.i = 0x8000000000000000LL - fa.i;
  // Make bInt lexicographically ordered as a twos-complement int
  dasi fb; fb.f = b;
  if (fb.i < 0) fb.i = 0x8000000000000000LL - fb.i;
  return std::abs(fa.i - fb.i);
}

template<typename T>
inline bool almostEqual(T a, T b, int maxUlps) {
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.
  // assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
  return intDiff(a,b) <= maxUlps;
}


namespace {
  template<typename T>
  inline T eta(T x, T y, T z) { T t(z/std::sqrt(x*x+y*y)); return std::log(t+std::sqrt(t*t+T(1)));} 
  template<typename T>
  inline T eta2(T x, T y, T z) { T t = (z*z)/(x*x+y*y); return copysign(std::log(std::sqrt(t)+std::sqrt(t+T(1))), z); }

  inline float eta3(float x, float y, float z) { float t(z/std::sqrt(x*x+y*y)); return ::asinhf(t);} 
  inline double eta3(double x, double y, double z) { double t(z/std::sqrt(x*x+y*y)); return ::asinh(t);} 

  
  void look(float x) {
    int e;
    float r = ::frexpf(x,&e);
    std::cout << x << " exp " << e << " res " << r << std::endl;
    
    union {
      float val;
      int bin;
    } f;
    
    f.val = x;
    printf("%e %a %x\n",  f.val,  f.val,  f.bin);
    // printf("%e %x\n", f.val,  f.bin);
    int log_2 = ((f.bin >> 23) & 255) - 127;  //exponent
    f.bin &= 0x7FFFFF;                               //mantissa (aka significand)
    
    std::cout << "exp " << log_2 << " mant in binary "  << std::hex << f.bin 
	      << " mant as float " <<  std::dec << (f.bin|0x800000)*::pow(2.,-23)
	      << std::endl << std::endl;
  }

  void look(double x) {
    // int e;
    // float r = ::frexpf(x,&e);
    // std::cout << x << " exp " << e << " res " << r << std::endl;
    
    union {
      double val;
      long long bin;
    } f;
    
    f.val = x;

    printf("%e %a %x\n",  f.val,  f.val,  f.bin);
    // printf("%e %x\n", f.val,  f.bin);

  }

}

void peta() {
  std::cout << "T t(z/std::sqrt(x*x+y*y)); return std::log(t+std::sqrt(t*t+T(1)));" << std::endl;
  {
    float 
      xn = 122.436f, yn = 10.7118f, zn = -1115.f;
    float etan = eta(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   float 
      xp = 122.436f, yp = 10.7118f, zp = 1115.f;
    float etap = eta(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

 {
    double 
      xn = 122.436, yn = 10.7118, zn = -1115.;
    double etan = eta(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   double 
      xp = 122.436, yp = 10.7118, zp = 1115.;
    double etap = eta(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

}


void peta2() {
  std::cout << "T t = (z*z)/(x*x+y*y); return copysign(std::log(std::sqrt(t)+std::sqrt(t+T(1))), z);"<< std::endl;
  {
    float 
      xn = 122.436f, yn = 10.7118f, zn = -1115.f;
    float etan = eta2(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   float 
      xp = 122.436f, yp = 10.7118f, zp = 1115.f;
    float etap = eta2(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

 {
    double 
      xn = 122.436, yn = 10.7118, zn = -1115.;
    double etan = eta2(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   double 
      xp = 122.436, yp = 10.7118, zp = 1115.;
    double etap = eta2(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

}

void peta3() {
  std::cout << "t(z/std::sqrt(x*x+y*y)); return ::asinh(t);" << std::endl;
  {
    float 
      xn = 122.436f, yn = 10.7118f, zn = -1115.f;
    float etan = eta3(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   float 
      xp = 122.436f, yp = 10.7118f, zp = 1115.f;
    float etap = eta3(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

 {
    double 
      xn = 122.436, yn = 10.7118, zn = -1115.;
    double etan = eta3(xn,yn,zn);
 
    std::cout << etan << std::endl;
    look(etan);
    std::cout << -etan << std::endl;
    look(-etan);

   double 
      xp = 122.436, yp = 10.7118, zp = 1115.;
    double etap = eta3(xp,yp,zp);
 

    std::cout << etap << std::endl;
    look(etap);

    std::cout << intDiff(etap,-etan) << std::endl << std::endl;

  }

}

int main() {

  peta();
  peta2();
  peta3();

    return 0;

}
