#include "DataFormats/Math/interface/SSEVec.h"
#include<cmath>
#include<iostream>

// this is a test,
using namespace mathSSE;

void addScaleddiff(Vec3F&res, float s, Vec3F const & a, Vec3F const & b) {
  res = res + s*(a-b);
} 

void addScaleddiffIntr(Vec3F&res, float s, Vec3F const & a, Vec3F const & b) {
  res.vec =  _mm_add_ps(res.vec, _mm_mul_ps(_mm_set1_ps(s), _mm_sub_ps(a.vec,b.vec)));
}


float dotV( Vec3F const & a, Vec3F const & b) {
  return dot(a,b);
}

float dotSimple( Vec3F const & a, Vec3F const & b) {
  Vec3F res = a*b;
  return res.arr[0]+res.arr[1]+res.arr[2];

}

float norm(Vec3F const & a) {
  return std::sqrt(dot(a,a));
}



// fake basicVector to check constructs...
template<typename T>
struct BaVec { 
  typedef BaVec<T> self;

  self & operator+=(self const & rh) {
    return *this;
  }

  T  theX; T  theY; T  theZ; T  theW;
}  __attribute__ ((aligned (16)));


typedef BaVec<float> BaVecF;

struct makeVec3F {
  makeVec3F(BaVecF & bv) : v(reinterpret_cast<Vec3F&>(bv)){}
  Vec3F & v;
};
struct makeVec3FC {
  makeVec3FC(BaVecF const & bv) : v(reinterpret_cast<Vec3F const&>(bv)){}
  Vec3F const & v;
};

template<>
inline BaVecF & BaVecF::operator+=(BaVecF const & rh) {
  makeVec3FC v(rh);
  makeVec3F s(*this);
  s.v = s.v + v.v;
  return *this;
}


void sum(BaVecF & lh, BaVecF const & rh) {
  lh += rh;  
}

int main() {

  Vec3F x(2.0,4.0,5.0);
  Vec3F y(-3.0,2.0,-5.0);

  std::cout << dot(x,y) << std::endl; 
  std::cout << dotSimple(x,y) << std::endl;

  Vec3F z = cross(x,y);
  std::cout << z.arr[0] << ", "<< z.arr[1] << ", "<< z.arr[2] << std::endl;
  std::cout << z.oldVec.theX << ", "<<  z.oldVec.theY << ", "<<  z.oldVec.theZ << std::endl;

}
