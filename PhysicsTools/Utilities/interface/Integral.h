#ifndef PhysicsTools_Utilities_Integral_h
#define PhysicsTools_Utilities_Integral_h
#include "PhysicsTools/Utilities/interface/Primitive.h"

namespace funct {

  struct no_var;

  template<typename F, typename X = no_var> 
  struct IntegralStruct {
    IntegralStruct(const F& f) : p(primitive<X>(f)) { }
    double operator()(double min, double max) const { 
      X::set(min); double pMin = p();
      X::set(max); double pMax = p();
      return pMax - pMin; 
    }
  private:
    typename Primitive<F, X>::type p;
  };

  template<typename F>
  struct IntegralStruct<F, no_var> {
    IntegralStruct(const F& f) : p(primitive(f)) { }
    double operator()(double min, double max) const { 
      return p(max) - p(min); 
    }
  private:
    typename Primitive<F>::type p;
  };

  template<unsigned samples, typename F, typename X = no_var> 
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

  template<unsigned samples, typename F>
  struct NumericalIntegral<samples, F, no_var> {
    NumericalIntegral(const F& f) : _f (f) { }
    double operator()(double min, double max) const { 
      double l = max - min;
      double delta = l / samples;
      double sum = 0;
      for(unsigned int i = 0; i < samples; i++) {
	double x = min + (i + 0.5) * delta;
	sum += _f(x);
      }
      return sum * l / samples;
    }
  private:
    F _f;
  };

  template<typename F, typename X = no_var> struct Integral {
    typedef IntegralStruct<F, X> type;
  };

  template<typename X, typename F> 
  double integral(const F& f, double min, double max) {
    typename Integral<F, X>::type i(f);
    return i(min, max);
  }

  template<typename F>
  double integral(const F& f, double min, double max) {
    typename Integral<F>::type i(f);
    return i(min, max);
  }
}

#define NUMERICAL_INTEGRAL(X, F, SAMPLES) \
namespace funct { \
  template<typename X> struct Integral<F, X> { \
    typedef NumericalIntegral<SAMPLES, F, X> type; \
  }; \
} \
struct __useless_ignoreme

#define NUMERICAL_FUNCT_INTEGRAL(F, SAMPLES) \
namespace funct { \
  template<> struct Integral<F, no_var> { \
    typedef NumericalIntegral<SAMPLES, F> type; \
  }; \
} \
struct __useless_ignoreme

#endif
