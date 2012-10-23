#include "FWCore/Utilities/interface/isFinite.h"

#include<limits>
#include<cmath>
#include<cstdlib>
#include<cassert>

int main(int n, const char**) {
  using namespace edm;
  
  
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
  assert(isNotFinite(1./zero) );
  assert(isNotFinite(float(1.)/float(zero)) );
  assert(isNotFinite(-1./zero) );
  assert(isNotFinite(-1.f/float(zero)) );

  return 0;
}

