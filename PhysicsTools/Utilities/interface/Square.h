#ifndef PhysicsTools_Utilities_Sqare_h
#define PhysicsTools_Utilities_Sqare_h
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Power.h"

namespace funct {

  template<typename F> struct Square { 
    typedef typename Power<F, Numerical<2> >::type type; 
  };

  template<typename F> 
  typename Square<F>::type sqr(const F& f) { 
    return pow(f, num<2>()); 
  }

}
#endif
