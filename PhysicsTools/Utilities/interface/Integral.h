#ifndef PhysicsTools_Utilities_Integral_h
#define PhysicsTools_Utilities_Integral_h
#include "PhysicsTools/Utilities/interface/Primitive.h"

namespace funct {

  template<typename X, typename F> 
  struct IntegralStruct {
    IntegralStruct(const F& f) : p(primitive<X>(f)) { }
    double operator()(double min, double max) const { 
      X::set(min); double pMin = p();
      X::set(max); double pMax = p();
      return pMax - pMin; 
    }
  private:
    typename Primitive<X, F>::type p;
  };

  template<typename X, typename F, unsigned samples> 
  struct NumericalIntegral {
    NumericalIntegral(const F& f) : _f (f) { }
    double operator()(double min, double max) const { 
      double l = max - min;
      double delta = l / samples;
      double sum = 0;
      for(unsigned int i = 0; i < samples; i++) {
	X::set(min + (i + 0.5) * delta);
	sum += _f();
      }
      return sum * l / samples;
    }
  private:
    F _f;
  };

  template<typename X, typename F> struct Integral {
    typedef IntegralStruct<X, F> type;
  };

  template<typename X, typename F> 
    double integral(const F& f, double min, double max) {
    typename Integral<X, F>::type i(f);
    return i(min, max);
  }
  
}

#define NUMERICAL_INTEGRAL(X, F, SAMPLES) \
namespace funct { \
  template<typename X> struct Integral<X, F> { \
    typedef NumericalIntegral<X, F, SAMPLES> type; \
  }; \
} \
struct __useless_ignoreme

#endif
