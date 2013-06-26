#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "DataFormats/Math/interface/approx_erf.h"
#include<cstdio>
#include<cstdlib>
#include<iostream>

void printEXP(float x) {
  printf("x= %a val= %a %a %a\n",x, std::exp(double(x)),unsafe_expf<6>(x),unsafe_expf<4>(x));  
  printf("x= %f val= %f %f %f\n",x, std::exp(double(x)),unsafe_expf<6>(x),unsafe_expf<4>(x));  
  printf("x= %f val= %f %f %f\n\n",x, std::exp(x),approx_expf<6>(x),approx_expf<4>(x));  
}

void printLOG(float x) {
  printf("x= %a val= %a %a %a\n",x, std::log(double(x)),unsafe_logf<8>(x),unsafe_logf<4>(x));  
  printf("x= %f val= %f %f %f\n",x, std::log(double(x)),unsafe_logf<8>(x),unsafe_logf<4>(x));  
  printf("x= %f val= %f %f %f\n\n",x, std::log(x),approx_logf<6>(x),approx_logf<4>(x));  
}


template<typename PRINT>
void printLim(PRINT printit) {
  constexpr float zero_threshold_ftz =-float(0x5.75628p4);
  constexpr float inf_threshold =float(0x5.8b90cp4);
  std::cout<< "\nexp valid for " << zero_threshold_ftz << " < x < "
     <<  inf_threshold  << std::endl;
  printit(zero_threshold_ftz);
  printit(zero_threshold_ftz-1);
  printit(zero_threshold_ftz+1);
  printit(4*zero_threshold_ftz);
  printit(0);
  printit(1.);
  printit(-1.);
  printit(inf_threshold);
  printit(inf_threshold+1);
  printit(inf_threshold-1);
  printit(4*inf_threshold);
  printit(std::sqrt(-1));
  printit(std::numeric_limits<float>::infinity());
  printit(-std::numeric_limits<float>::infinity());

  std::cout<< "\n" << std::endl;
}

namespace justcomp {
  constexpr int NN=1024*1024;
  float a[NN], b[NN];
  template<int DEGREE>
  void bar() {
    for (int i=0; i!=NN; ++i)
      b[i] = approx_expf<DEGREE>(a[i]);
  }
  template<int DEGREE>
  void foo() {
    for (int i=0; i!=NN; ++i)
      b[i] = unsafe_expf<DEGREE>(a[i]);
  }
  template<int DEGREE>
  void lar() {
    for (int i=0; i!=NN; ++i)
      b[i] = approx_logf<DEGREE>(a[i]);
  }
  template<int DEGREE>
  void loo() {
    for (int i=0; i!=NN; ++i)
      b[i] = unsafe_logf<DEGREE>(a[i]);
  }

}


template<typename STD, typename APPROX>
void accTest(STD stdf, APPROX approx, int degree) {
  using namespace approx_math;
  std::cout << std::endl << "launching  exhaustive test for degree " << degree << std::endl;
  binary32 x,r,ref;
  int maxdiff=0;
  int n127=0;
  int n16393=0;
  x.ui32=0; // should be 0 but 
  while(x.ui32<0xffffffff) {
    x.ui32++;
    // remove nans..
    if ( (x.ui32&0x7f80000) && x.ui32&0x7FFFFF) continue;
    r.f=approx(x.f);
    ref.f=stdf(double(x.f)); // double-prec one  (no hope with -fno-math-errno)
    int d=abs(r.i32-ref.i32);
    if(d>maxdiff) {
      // std::cout << "new maxdiff for x=" << x.f << " : " << d << std::endl;
      maxdiff=d;
	}
    if (d>127) ++n127;
    if (d>16393) ++n16393;
  }
  std::cout << "maxdiff / diff >127 / diff >16393 " << maxdiff << " / " << n127<< " / " << n16393<< std::endl;
}

template<typename STD, typename APPROX>
void accuTest(STD stdf, APPROX approx, const char * name, float mm=std::numeric_limits<float>::min(), float mx=std::numeric_limits<float>::max()) {
  using namespace approx_math;
  std::cout << std::endl << "launching  exhaustive test for " << name << std::endl;
  binary32 x,pend, r,ref;
  int maxdiff=0;
  int n127=0;
  int n16393=0;
  float ad=0., rd=0;
  x.f=mm;
  x.ui32++;
  pend.f=mx;
  pend.ui32--;
  std::cout << "limits " << x.f << " " << pend.f << " " << pend.ui32-x.ui32 << std::endl;
  while(x.ui32<pend.ui32) {
    x.ui32++;
    r.f=approx(x.f);
    ref.f=stdf(x.f); // double-prec one  (no hope with -fno-math-errno)
    ad = std::max(ad,std::abs(r.f-ref.f));
    rd = std::max(rd,std::abs((r.f/ref.f)-1.f));
    int d=abs(r.i32-ref.i32);
    if(d>maxdiff) {
      // std::cout << "new maxdiff for x=" << x.f << " : " << d << std::endl;
      maxdiff=d;
	}
    if (d>127) ++n127;
    if (d>16393) ++n16393;
  }
  std::cout << "absdiff / reldeff/ maxdiff / diff >127 / diff >16393 :  " << ad << " / " << rd << " / "  
	    << maxdiff << " / " << n127<< " / " << n16393<< std::endl;
}




// performance test
#ifndef __arm__
#include <x86intrin.h>
#include <cpuid.h>
#ifdef __clang__
bool has_rdtscp() { return true;}
/** CPU cycles since processor startup */
inline uint64_t rdtsc() {
uint32_t lo, hi;
/* We cannot use "=A", since this would use %rax on x86_64 */
__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
return (uint64_t)hi << 32 | lo;
}
#else
// CPUID, EAX = 0x80000001, EDX values
#ifndef bit_RDTSCP
#define bit_RDTSCP          (1 << 27)
#endif
namespace {
  inline
  bool has_rdtscp() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0x80000001, & eax, & ebx, & ecx, & edx))
      return (edx & bit_RDTSCP) != 0;
    else
      return false;
  }
  unsigned int rdtscp_val=0;
  inline volatile unsigned long long rdtsc() {
    return __rdtscp(&rdtscp_val);
  }
}
#endif
#else  // arm
namespace {
inline bool has_rdtscp() { return false;}
inline volatile unsigned long long rdtsc() {return 0;}
}
#endif // arm


template<int DEGREE, int WHAT>
struct Measure{
  inline void operator()(unsigned long long & t) const;
};

template<int DEGREE>
struct Measure<DEGREE,0> {
  inline
    void operator()(unsigned long long & t) const{
    t -= rdtsc();
    justcomp::foo<DEGREE>();
    t += rdtsc();
  }
};


template<int DEGREE>
struct Measure<DEGREE,1> {
inline
void operator()(unsigned long long & t) const{
  t -= rdtsc();
  justcomp::bar<DEGREE>();
  t += rdtsc();
 }
};

template<int DEGREE>
struct Measure<DEGREE,2> {
inline
void operator()(unsigned long long & t) const{
  t -= rdtsc();
  justcomp::loo<DEGREE>();
  t += rdtsc();
 }
};

template<int DEGREE>
struct Measure<DEGREE,3> {
inline
void operator()(unsigned long long & t) const{
  t -= rdtsc();
  justcomp::lar<DEGREE>();
  t += rdtsc();
 }
};


template<int DEGREE, int WHAT=1>
void perf() {
  if (!has_rdtscp()) return; 
  Measure<DEGREE,WHAT> measure;
  using namespace approx_math;
  unsigned long long t=0;
  binary32 x,r;
  float sum=0;
  long long ntot=0;
  x.f=1.0; // should be 0 but 
  while (x.f<32) { // this is 5*2^23 tests
    ++ntot;
    int i=0;
    while(i<justcomp::NN) { 
      x.ui32++;
      justcomp::a[i++]=x.f;
      justcomp::a[i++]= (WHAT<2) ? -x.f : 1.f/x.f;
    }
    measure(t);
    //  r.f=approx_expf<6>(x.f);// time	0m1.180s
    // r.f=expf(x.f);	// time 0m4.372s
    // r.f=exp(x.f);  // time 	0m1.789s
    for (int i=0; i!=justcomp::NN; ++i)
      sum += justcomp::b[i];
  }
  const char * what[]={"exp unsafe","exp apporx","log unsafe","log approx"};
  std::cout << "time for " << what[WHAT] << " degree " << DEGREE << " is "<< double(t)/double(justcomp::NN*ntot) << std::endl;
  std::cout << "sum= " << sum << " to prevent compiler optim." << std::endl;;
  
}


int main() {
  printLim(printEXP);
  printLim(printLOG);

  perf<2,0>();
  perf<3,0>();
  perf<3>();
  perf<4>();
  perf<6>();

  perf<2,2>();
  perf<2,3>();
  perf<4,2>();
  perf<4,3>();
  perf<8,2>();
  perf<8,3>();


  accTest(::exp,approx_expf<2>,2);
  accTest(::exp,approx_expf<3>,3);
  accTest(::exp,approx_expf<4>,4);
  accTest(::exp,approx_expf<6>,6);


  accTest(::log,approx_logf<2>,2);
  accTest(::log,approx_logf<4>,4);
  accTest(::log,approx_logf<8>,8);


  accuTest(::erf,approx_erf, "erf", .01, 8  );


  return 0;
}
