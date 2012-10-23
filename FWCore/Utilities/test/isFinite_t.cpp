#include "FWCore/Utilities/interface/isFinite.h"

#include<limits>
#include<cmath>
#include<cstdlib>
#include<cassert>

int main(int n, const char**) {
  using namespace edm;
  typedef long double LD;
  
  double zero = atof("0");
  
  assert(isFinite(double(0)) );
  assert(isFinite(float(0)) );
  assert(isFinite(double(-3.14)) );
  assert(isFinite(float(-3.14)) );
  assert(!isFinite(std::sqrt(-double(n))) );
  assert(!isFinite(std::sqrt(-float(n))) );
  assert(!isFinite(1./zero) );
  assert(!isFinite(float(1.)/float(zero)) );
  assert(!isFinite(-1./zero) );
  assert(!isFinite(-1.f/float(zero)) );

  //

  assert(!isNotFinite(double(0)) );
  assert(!isNotFinite(float(0)) );
  assert(!isNotFinite(double(-3.14)) );
  assert(!isNotFinite(float(-3.14)) );
  assert(isNotFinite(std::sqrt(-double(n))) );
  assert(isNotFinite(std::sqrt(-float(n))) );
  assert(isNotFinite(-1.f/float(zero)) );
  assert(isNotFinite(float(1.)/float(zero)) );
  assert(isNotFinite(-1./zero) );
  assert(isNotFinite(-1.f/float(zero)) );

  assert(!isNotFinite(LD(3.14)) );
  assert(isNotFinite(-1/LD(zero)) );
  assert(isNotFinite(std::sqrt(-LD(n))) );

  return 0;
}

