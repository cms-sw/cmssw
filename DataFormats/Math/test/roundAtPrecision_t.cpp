#include "DataFormats/Math/interface/roundAtPrecision.h"

#include <cmath>

#include<cassert>


int main() {



  float almost05m = std::nextafter(0.5f,0.f);
  float almost05p = std::nextafter(0.5f,1.f);

  assert(roundAtPrecision<8>(almost05m)==0.5f);
  assert(roundAtPrecision<8>(almost05p)==0.5f);
  assert(roundAtPrecision<8>(-almost05m)==-0.5f);
  assert(roundAtPrecision<8>(-almost05p)==-0.5f);

  assert(roundAtPrecision<0>(M_PI)==4.f);
  assert(roundAtPrecision<0>(2.9)==2.f);
  assert(roundAtPrecision<0>(-M_PI)==-4.f);
  assert(roundAtPrecision<0>(-2.9)==-2.f);


  assert(roundAtPrecision<23>(M_PI)==float(M_PI));
  assert(roundAtPrecision<23>(-M_PI)==float(-M_PI));



  return 0;

}

