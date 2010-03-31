#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include<iostream>

// this is a test,
// using namespace mathSSE;

void addScaleddiff(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res += s*(a-b);
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

  Basic3DVectorF  x(2.0,4.0,5.0);
  Basic3DVectorF  y(-3.0,2.0,-5.0);

  std::cout << dotV(x,y) << std::endl; 
  std::cout << normV(x) << std::endl; 
  std::cout << norm(x) << std::endl; 


  Basic3DVectorF  z = x.cross(y);
  std::cout << z << std::endl;

  Basic3DVectorF  vx(2.0,4.0,5.0);
  Basic3DVectorF  vy(-3.0,2.0,-5.0);
  vx+=vy;
  std::cout << vx << std::endl;

  Basic3DVectorF z(1.,1.,1);
  addScaleddiff(z,0.1,vx,vy);
  std::cout << z << std::endl;

}
