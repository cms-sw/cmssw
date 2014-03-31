#include "DataFormats/Math/interface/approx_atan2.h"


#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<cassert>

#include "rdtscp.h"


namespace {
template <typename T> 
inline T toPhi (T phi) { 
  T result = phi;
  while (result > T(M_PI)) result -= T(2*M_PI);
  while (result <= -T(M_PI)) result += T(2*M_PI);
  return result;
}


template <typename T> 
inline T deltaPhi (T phi1, T phi2) { 
  T result = phi1 - phi2;
  while (result > T(M_PI)) result -= T(2*M_PI);
  while (result <= -T(M_PI)) result += T(2*M_PI);
  return result;
}


inline bool phiLess(float x, float y) {
  auto ix = phi2int(toPhi(x));
  auto iy = phi2int(toPhi(y));

  return (ix-iy)<0;

}

}


template<> float unsafe_atan2f<0>(float,float) { return 1.f;}
template<> float unsafe_atan2f<99>(float y,float x) { return std::atan2(y,x);}
template<> int unsafe_atan2i<0>(float,float) { return 1;}
template<> int unsafe_atan2i<99>(float y,float x) { return phi2int(std::atan2(y,x));}



template<int DEGREE>
void diff() {
  float mdiff=0;
  int idiff=0;
  constexpr float xmin=-100.001;  // avoid 0
  constexpr float incr = 0.04;
  constexpr int N = 2.*std::abs(xmin)/incr;
  auto y = xmin;
  for (int i=0; i<N; ++i) {
    auto x=xmin;
    for (int i=0; i<N; ++i) {
      auto approx = unsafe_atan2f<DEGREE>(y,x);
      auto iapprox = unsafe_atan2i<DEGREE>(y,x);
      auto std = std::atan2(y,x);
      mdiff = std::max(std::abs(std-approx),mdiff);
      idiff = std::max(std::abs(phi2int(std)-iapprox),idiff);
      x+=incr;
    }
    y+=incr;
  }
  
  std::cout << "for degree " << DEGREE << " max diff is " << mdiff << ' ' << idiff << ' ' << int2phi(idiff) <<  std::endl; 

}


template<int DEGREE>
void speedf() {
  if(!has_rdtscp()) return;
  long long t = -rdtsc();
  float approx=0;
  constexpr float xmin=-400.001;  // avoid 0
  constexpr float incr = 0.02;
  constexpr int N = 2.*std::abs(xmin)/incr;
  auto y = xmin;
  for (int i=0; i<N; ++i) {
    auto x=xmin;
    for (int i=0; i<N; ++i) {
      approx+= unsafe_atan2f<DEGREE>(y,x);
      x+=incr;
    }
    y+=incr;
  }
  t +=rdtsc();

  std::cout << "f for degree " << DEGREE << " clock is " << t << " " << approx << std::endl;
}

template<int DEGREE>
void speeds() {
  if(!has_rdtscp()) return;
  long long t = -rdtsc();
  float approx=0;
  constexpr float xmin=-400.001;  // avoid 0
  constexpr float incr = 0.02;
  constexpr int N = 2.*std::abs(xmin)/incr;
  auto y = xmin;
  for (int i=0; i<N; ++i) {
    auto x=xmin;
    for (int i=0; i<N; ++i) {
      approx+= safe_atan2f<DEGREE>(y,x);
      x+=incr;
    }
    y+=incr;
  }
  t +=rdtsc();

  std::cout << "s for degree " << DEGREE << " clock is " << t << " " << approx << std::endl;
}


template<int DEGREE>
void speedi() {
  if(!has_rdtscp()) return;
  long long t = -rdtsc();
  int approx=0;
  constexpr float xmin=-400.001;  // avoid 0
  constexpr float incr = 0.02;
  constexpr int N = 2.*std::abs(xmin)/incr;
  auto y = xmin;
  for (int i=0; i<N; ++i) {
    auto x=xmin;
    for (int i=0; i<N; ++i) {
      approx+= unsafe_atan2i<DEGREE>(y,x);
      x+=incr;
    }
    y+=incr;
  }
  t +=rdtsc();

  std::cout << "i for degree " << DEGREE << " clock is " << t << " " << approx << std::endl;
}



void testIntPhi() {

  constexpr long long maxint = (long long)(std::numeric_limits<int>::max())+1LL;
  constexpr int pi2 =  int(maxint/2LL);
  constexpr int pi4 =  int(maxint/4LL);
  constexpr int pi34 = int(3LL*maxint/4LL);

  std::cout << "pi,  pi2,  pi4, p34 " << maxint << ' ' << pi2 << ' ' << pi4 << ' ' << pi34 << ' ' << pi2+pi4  << '\n';
  std::cout << "Maximum value for int: " << std::numeric_limits<int>::max() << '\n';
  std::cout << "Maximum value for int+2: " << std::numeric_limits<int>::max()+2 << '\n';
  std::cout << "Maximum value for int+1 as LL: " << (long long)(std::numeric_limits<int>::max())+1LL << std::endl;

  std::cout << "Maximum value for short: " << std::numeric_limits<short>::max() << '\n';
  std::cout << "Maximum value for short+2: " << short(std::numeric_limits<short>::max()+short(2)) << '\n';
  std::cout << "Maximum value for short+1 as int: " << (int)(std::numeric_limits<short>::max())+1 << std::endl;


  auto d = float(M_PI) -std::nextafter(float(M_PI),0.f);
  std::cout << "abs res at pi for float " << d << ' ' << phi2int(d) << std::endl;
  std::cout << "abs res at for int " << int2dphi(1) << std::endl;
  std::cout << "abs res at for short " << short2phi(1) << std::endl;


  assert(-std::numeric_limits<int>::max() == (std::numeric_limits<int>::max()+2));

  assert(phiLess(0.f,2.f));
  assert(phiLess(6.f,0.f));
  assert(phiLess(3.2f,0.f));
  assert(phiLess(3.0f,3.2f));

  assert(phiLess(-0.3f,0.f));
  assert(phiLess(-0.3f,0.1f));
  assert(phiLess(-3.0f,0.f));
  assert(phiLess(3.0f,-3.0f));
  assert(phiLess(0.f,-3.4f));

  // go around the clock
  constexpr float eps = 1.e-5;
  auto ok = [](float a, float b) { assert(std::abs(a-b)<eps);};
  float phi1= -7.;
  while (phi1<8) {
    auto p1 = toPhi(phi1);
    auto ip1 = phi2int(p1);
    std::cout << "phi1 " << phi1 << ' ' << p1 << ' ' << ip1 << ' ' << int2phi(ip1) << std::endl;
    ok(p1,int2phi(ip1));
    float phi2= -7.2;
    while (phi2<8) {
    auto p2 = toPhi(phi2);
    auto ip2 = phi2int(p2);
    std::cout << "phi2 " << phi2 << ' ' <<  deltaPhi(phi1,phi2)  << ' ' <<  deltaPhi(phi2,phi1)
	      << ' ' << int2phi(ip1-ip2) << ' ' << int2phi(ip2-ip1)   
	      << ' ' <<  toPhi(phi2+phi1) << ' ' << int2phi(ip1+ip2) << std::endl;
    ok(deltaPhi(phi1,phi2),int2phi(ip1-ip2));
    ok(deltaPhi(phi2,phi1),int2phi(ip2-ip1));
    ok(toPhi(phi2+phi1),int2phi(ip1+ip2));
      phi2+=1;
    }

    phi1+=1;
  }

}

int main() {

  std::cout << unsafe_atan2f<5>(0.f,0.f) << " " << std::atan2(0.,0.) << std::endl;
  std::cout << unsafe_atan2f<5>(0.5f,0.5f) << " " << std::atan2(0.5,0.5) << std::endl;
  std::cout << unsafe_atan2f<5>(0.5f,-0.5f) << " " << std::atan2(0.5,-0.5) << std::endl;
  std::cout << unsafe_atan2f<5>(-0.5f,-0.5f) << " " << std::atan2(-0.5,-0.5) << std::endl;
  std::cout << unsafe_atan2f<5>(-0.5f,0.5f) << " " << std::atan2(-0.5,0.5) << std::endl;

  std::cout << safe_atan2f<15>(0.f,0.f) << " " << std::atan2(0.,0.) << std::endl;
  std::cout << safe_atan2f<15>(0.5f,0.5f) << " " << std::atan2(0.5,0.5) << std::endl;
  std::cout << safe_atan2f<15>(0.5f,-0.5f) << " " << std::atan2(0.5,-0.5) << std::endl;
  std::cout << safe_atan2f<15>(-0.5f,-0.5f) << " " << std::atan2(-0.5,-0.5) << std::endl;
  std::cout << safe_atan2f<15>(-0.5f,0.5f) << " " << std::atan2(-0.5,0.5) << std::endl;

  testIntPhi();



  diff<3>();
  diff<5>();
  diff<7>();
  diff<9>();
  diff<11>();
  diff<13>();
  diff<15>();
  diff<99>();


  speedf<0>();
  speedf<3>();
  speedf<5>();
  speedf<7>();
  speedf<9>();
  speedf<11>();
  speedf<13>();
  speedf<15>();
  speedf<99>();

  speeds<5>();
  speeds<11>();
  speeds<15>();
 

  speedi<0>();
  speedi<3>();
  speedi<5>();
  speedi<7>();
  speedi<9>();
  speedi<11>();
  speedi<13>();
  speedi<99>();


  return 0;


}
