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
#ifdef NO_LIKELY
  std::cout << "NO_LIKELY" << std::endl;
#endif
#ifdef REVERSE_LIKELY
  std::cout << "REVERSE_LIKELY" << std::endl;
#endif

  assert(!test(0));
  assert(test(2));

  return 0;
}
