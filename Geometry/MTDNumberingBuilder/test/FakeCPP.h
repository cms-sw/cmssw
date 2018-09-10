/*
 * Fake cpp_unit to be used if cpp_unit is not avaliable
 *
 */
#include <utility>
#include <iostream>

namespace cppUnit {

   std::pair<int,int> & stats() {
    static std::pair<int,int> passedFailed(0,0);
    return  passedFailed;
  }

  bool test(bool pf) {
    if (pf) stats().first++;
    else stats().second++;
    return pf;
  }

  struct Dump {
    Dump(){}
    ~Dump(){
      std::cerr << "Test passed: " << stats().first << std::endl;
      std::cerr << "Test failed: " << stats().second << std::endl;
    }
  };

  
}

#define CPPUNIT_ASSERT(x) if (!cppUnit::test(x)) std::cerr<< "failed  "<< #x << std::endl;

