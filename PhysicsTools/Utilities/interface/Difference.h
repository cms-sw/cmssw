#ifndef PhysicsTools_Utilities_Difference_h
#define PhysicsTools_Utilities_Difference_h
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Minus.h"

namespace funct {

  template<typename A, typename B>
  struct Difference { 
    typedef typename Sum<A, typename Minus<B>::type>::type type; 
    inline static type combine(const A& a, const B& b) { return a + (-b); }
  };

  template<typename A, typename B>
  inline typename Difference<A, B>::type operator-(const A& a, const B& b) {
    return Difference<A, B>::combine(a, b);
  }

}

#endif
