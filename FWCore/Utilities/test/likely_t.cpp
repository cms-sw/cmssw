#include "FWCore/Utilities/interface/Likely.h"
#include <iostream>
#include <cassert>

// test that compiles and does not interfere with the logic...
namespace {
  bool test(int n) {
    bool ret=true;
    if (likely(n>1)) ret&=true; 
    else
      ret=false;
    
    if (unlikely(n>1)) ret&=true;
    else
      ret =false;

    ret &=likely(n>1);
    ret &=unlikely(n>1);
    return ret;
  }
}


int main() {

  assert(!test(0));
  assert(test(2));

  return 0;
}
