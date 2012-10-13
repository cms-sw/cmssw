#include<cmath>
inline float __attribute__((always_inline)) __attribute__ ((pure))
eta(float x, float y, float z) { float t(z/std::sqrt(x*x+y*y)); return ::asinhf(t);} 

struct Vector {
  Vector(){}
  Vector(float ia, float ib, float ic) : a(ia),b(ib),c(ic){}
  float a,b,c;

  float x() const { return a;}
  float y() const { return b;}
  float z() const { return c;}
  float phi() const { return std::atan2(y(),x());}
  float perp2() const { return a*a+b*b;}
  float eta() const { return ::eta(a,b,c);}

};


#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Math/interface/approx_log.h"

inline
int diff(float a, float b) {
  approx_math::binary32 ba(a);
  approx_math::binary32 bb(b);
  return ba.i32 - bb.i32;

}

#include<cstdio>
#include<iostream>
#include<vector>
int main() {
 std::vector<Vector> vs;
   for (float x=-1000.; x<=1010.; x+=100) 
      for (float y=-1000.; y<=1010.; y+=100) 
	for (float z=-1000.; z<=1010.; z+=100)
	  vs.emplace_back(x,y,z);
   std::cout << "testing " << vs.size() << " vectors" << std::endl;

   for (auto v1: vs)
     for (auto v2: vs) {
       float drv = reco::deltaR2(v1,v2);
       float dro = reco::deltaR2(v1.eta(),v1.phi(),v2.eta(),v2.phi());
       if ( std::abs(diff(drv,dro))>1 ) printf("%d %a %a\n",diff(drv,dro),drv,dro); 
     }
       

  return 0;
}
