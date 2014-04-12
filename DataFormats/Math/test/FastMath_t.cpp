#include "DataFormats/Math/interface/FastMath.h"

#include<iostream>
// #include <pmmintrin.h>
#include<typeinfo>

#include "FWCore/Utilities/interface/HRRealTime.h"

namespace {
  
  template<typename T> 
  std::pair<T,T> stdatan2r(T x, T y) {
    return std::pair<T,T>(std::atan2(x,y),std::sqrt(x*x+y*y));
  }
  
  template<typename T>
  struct Stat {
    std::string name;
    size_t n;
    size_t npos;
    T bias;
    T ave;
    T rms;
    T amax;
    Stat( std::string in): name(in), n(0),npos(0), bias(0),ave(0),rms(0), amax(0){}
    void operator()(T x, T ref);

    ~Stat() {
      std::cout << name << " "
		<< n << " " << npos << " " << bias/n << " " << ave/n 
		<< " " << (n*rms-bias*bias)/(n*(n-1))
		<< " " << amax << std::endl;
    }
  };
  
 template<typename T>
 void Stat<T>::operator()(T x, T ref) {
      n++;
      if (x>ref) npos++;
      T d = (x-ref)/std::abs(ref);
      bias += d;
      ave +=std::abs(d);
      rms +=d*d;
      amax = std::max(amax,std::abs(d));
    }


  volatile double dummy;
  template<typename T> 
  inline T eta(T x, T y, T z) { x = z/std::sqrt(x*x+y*y); return std::log(x+std::sqrt(x*x+T(1)));}

  
  template<typename T> 
  void sampleSquare() {
    edm::HRTimeType tf=0;
    edm::HRTimeType ts=0;
    Stat<T> stata("atan2");
    Stat<T> statr("r");
    T fac[8] = {-8, -5., -2., -1., 1.,2.,5.,8.};
    for (int k=0;k<100;k++)
      for (T x = 1e-15; x<1.1e+15; x *=10)
	for (T y = 1e-15; y<1.1e+15; y *=10) 
	  for (int i=0;i!=8; ++i)
	    for (int j=0;j!=8; ++j) {
	      T xx = x*fac[i];
	      T yy = y*fac[j];
	      edm::HRTimeType sf = edm::hrRealTime();
	      std::pair<T,T> res = fastmath::atan2r(xx,yy);
	      tf += (edm::hrRealTime() -sf);
	      for (int l=0; l<i+j; ++l) dummy+=yy; // add a bit of random instruction
	      edm::HRTimeType ss = edm::hrRealTime();
	      std::pair<T,T> ref = stdatan2r(xx,yy);
	      ts += (edm::hrRealTime() -ss);
	      stata(res.first,ref.first);
	      statr(res.second,ref.second);
	    }
    std::cout << typeid(T).name() << " times " << tf << " " << ts << std::endl;
  }

 
}




int main() {
  // _mm_setcsr (_mm_getcsr () | 0x8040);    // on Intel, treat denormals as zero for full speed
  
  {
    std::pair<double, double> res = fastmath::atan2r(-3.,4.);
    std::cout << res.first << " " << res.second << std::endl;
    std::cout << atan2(-3.,4.) << std::endl;
  }
  {
    std::pair<double, double> res = fastmath::etaphi(-3.,4.,5.);
    std::cout << res.first << " " << res.second << std::endl;
    std::cout << eta(-3.,4.,5.)  << " " << std::atan2(4.,-3.) << std::endl;
  }
  
  
  {
    std::pair<float, float> res = fastmath::atan2r(3.f,-4.f);
    std::cout << res.first << " " << res.second << std::endl;
    std::cout << atan2f(3.f,-4.f) << std::endl;
    
  }
  {
    std::pair<double, double> res = fastmath::etaphi(3.f,-4.f,-5.f);
    std::cout << res.first << " " << res.second << std::endl;
    std::cout << eta(3.f,-4.f,-5.f)   << " " << std::atan2(-4.f,3.f) << std::endl;
  }

 
 
  sampleSquare<float>();
  sampleSquare<double>();
  return 0;

}
