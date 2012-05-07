#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include<vector>

#include<iostream>

// this is a test,
// using namespace mathSSE;

void addScaleddiff(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res += s*(a-b);
} 

void addScaleddiff(Basic3DVectorD&res, float s,  Basic3DVectorD const & a, Basic3DVectorD const & b) {
  res += s*(a-b);
} 

void multiSum(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res = s*(a-b) + s*(a+b);
} 

void multiSum(Basic3DVectorD&res, float s,  Basic3DVectorD const & a, Basic3DVectorD const & b) {
  res = s*(a-b) + s*(a+b);
} 





float dotV(  Basic3DVectorF const & a,  Basic3DVectorF const & b) {
  return a*b;
}


float norm(Basic3DVectorF const & a) {
  return std::sqrt(a*a);
}

float normV(Basic3DVectorF const & a) {
  return a.mag();
}


int main() {
#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)
  std::cout << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#endif
#ifdef USE_SSEVECT
  std::cout << "sse vector enabled in cmssw" << std::endl;
#endif

  std::cout << sizeof(Basic3DVectorF) << std::endl;
  std::cout << sizeof(Basic3DVectorD) << std::endl;

  Basic3DVectorF  x(2.0f,4.0f,5.0f);
  Basic3DVectorF  y(-3.0f,2.0f,-5.0f);

  std::cout << dotV(x,y) << std::endl; 
  std::cout << normV(x) << std::endl; 
  std::cout << norm(x) << std::endl; 


  Basic3DVectorF  z = x.cross(y);
  std::cout << z << std::endl;
  std::cout << -z << std::endl;

  {
    std::cout << "f" << std::endl;
    Basic3DVectorF  vx(2.0f,4.0f,5.0f);
    Basic3DVectorF  vy(-3.0f,2.0f,-5.0f);
    vx+=vy;
    std::cout << vx << std::endl;
    
    Basic3DVectorF vz(1.f,1.f,1.f);
    addScaleddiff(vz,0.1f,vx,vy);
    std::cout << vz << std::endl;
  }

 {
    std::cout << "d" << std::endl;
    Basic3DVectorD  vx(2.0,4.0,5.0);
    Basic3DVectorD  vy(-3.0,2.0,-5.0);
    vx+=vy;
    std::cout << vx << std::endl;
    
    Basic3DVectorD vz(1.,1.,1);
    addScaleddiff(vz,0.1,vx,vy);
    std::cout << vz << std::endl;
 }

 std::cout << "std::vector" << std::endl;
 std::vector<Basic3DVectorF> vec1; vec1.reserve(50);
 std::vector<float> vecf(21);
 std::vector<Basic3DVectorF> vec2(51);
 std::vector<Basic3DVectorF> vec3; vec3.reserve(23456);
}
