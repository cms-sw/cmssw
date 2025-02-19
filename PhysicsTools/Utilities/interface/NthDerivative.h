#ifndef PhysicsTools_Utilities_NthDerivative_h
#define PhysicsTools_Utilities_NthDerivative_h

#include "PhysicsTools/Utilities/interface/Derivative.h"

namespace funct {

  template<unsigned n, typename X, typename F>
    struct NthDerivative {
      typedef typename Derivative<X, typename NthDerivative<n - 1, X, F>::type>::type type;
      inline static type get(const F& f) { 
	return derivative<X>(NthDerivative< n - 1, X, F>::get(f)); 
      }
    };

  template<typename X, typename F>
    struct NthDerivative<1, X, F> : public Derivative< X, F > { };

  template<typename X, typename F> struct NthDerivative<0, X, F> {
    typedef F type;
    inline static type get(const F& f) { return f; }
  };
  
  template<unsigned n, typename X, typename F>
    typename NthDerivative<n, X, F>::type
    nth_derivative(const F& f) { return NthDerivative< n, X, F >::get(f); }
  
}

#endif
