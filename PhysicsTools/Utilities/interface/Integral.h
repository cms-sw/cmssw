#ifndef PhysicsTools_Utilities_Integral_h
#define PhysicsTools_Utilities_Integral_h
#include "PhysicsTools/Utilities/interface/Primitive.h"
#include "PhysicsTools/Utilities/interface/NumericalIntegration.h"

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
  struct IntegralStruct<F> {
    IntegralStruct(const F& f) : p(primitive(f)) { }
    double operator()(double min, double max) const { 
      return p(max) - p(min); 
    }
  private:
    typename Primitive<F>::type p;
  };

  template<typename Integrator, typename F, typename X = no_var> 
  struct NumericalIntegral {
    NumericalIntegral(const F& f, const Integrator & integrator) : 
      f_(f), integrator_(integrator) { }
    inline double operator()(double min, double max) const { 
      return integrator_(f_, min, max);
    }
  private:
    struct function {
      function(const F & f) : f_(f) { }
      inline double operator()(double x) const {
	X::set(x); return f_();
      }
    private:
      F f_;
    };
    function f_;
    Integrator integrator_;
  };

  template<typename Integrator, typename F>
  struct NumericalIntegral<Integrator, F, no_var> {
    NumericalIntegral(const F& f, const Integrator & integrator) : 
      f_(f), integrator_(integrator) { }
    double operator()(double min, double max) const { 
      return integrator_(f_, min, max);
    }
  private:
    F f_;
    Integrator integrator_;
  };

  template<typename F, typename X = no_var> struct Integral {
    typedef IntegralStruct<F, X> type;
  };

  template<typename X, typename F> 
  typename Integral<F, X>::type integral(const F& f) {
    return typename Integral<F, X>::type(f);
  }

  template<typename X, typename F, typename Integrator> 
  typename Integral<F, X>::type integral(const F& f, const Integrator & integrator) {
    return typename Integral<F, X>::type(f, integrator);
  }

   template<typename F, typename Integrator> 
  typename Integral<F>::type integral_f(const F& f, const Integrator & integrator) {
    return typename Integral<F>::type(f, integrator);
  }

  template<typename X, typename F> 
  double integral(const F& f, double min, double max) {
    return integral<X>(f)(min, max);
  }

  template<typename X, typename F, typename Integrator> 
  double integral(const F& f, double min, double max, const Integrator & integrator) {
    return integral<X>(f, integrator)(min, max);
  }

  template<typename F> 
  typename Integral<F>::type integral_f(const F& f) {
    return typename Integral<F>::type(f);
  }

 template<typename F>
  double integral_f(const F& f, double min, double max) {
    return integral_f(f)(min, max);
  }

  template<typename F, typename Integrator>
  double integral_f(const F& f, double min, double max, const Integrator & integrator) {
    return integral_f(f, integrator)(min, max);
  }

  template<typename F, typename MIN, typename MAX, typename Integrator = no_var, typename X = no_var>
  struct DefIntegral {
    DefIntegral(const F & f, const MIN & min, const MAX & max, const Integrator & integrator) : 
      f_(f), min_(min), max_(max), integrator_(integrator) { } 
    double operator()() const {
      return integral<X>(f_, min_(), max_(), integrator_);
    }
  private:
    F f_;
    MIN min_;
    MAX max_;
    Integrator integrator_;
  };

 template<typename F, typename MIN, typename MAX, typename X>
  struct DefIntegral<F, MIN, MAX, no_var, X> {
    DefIntegral(const F & f, const MIN & min, const MAX & max) : 
      f_(f), min_(min), max_(max) { } 
    double operator()(double x) const {
      return integral<X>(f_, min_(x), max_(x));
    }
  private:
    F f_;
    MIN min_;
    MAX max_;
  };

 template<typename F, typename MIN, typename MAX, typename Integrator>
  struct DefIntegral<F, MIN, MAX, Integrator, no_var> {
    DefIntegral(const F & f, const MIN & min, const MAX & max, const Integrator & integrator) : 
      f_(f), min_(min), max_(max), integrator_(integrator) { } 
    double operator()(double x) const {
      return integral_f(f_, min_(x), max_(x), integrator_);
    }
  private:
    F f_;
    MIN min_;
    MAX max_;
    Integrator integrator_;
  };

  template<typename F, typename MIN, typename MAX>
  struct DefIntegral<F, MIN, MAX, no_var, no_var> {
    DefIntegral(const F & f, const MIN & min, const MAX & max) : f_(f), min_(min), max_(max) { } 
    double operator()(double x) const {
      return integral_f(f_, min_(x), max_(x));
    }
  private:
    F f_;
    MIN min_;
    MAX max_;
  };

 }

#define NUMERICAL_INTEGRAL(X, F, INTEGRATOR) \
namespace funct { \
  template<typename X> struct Integral<F, X> { \
    typedef NumericalIntegral<INTEGRATOR, F, X> type; \
  }; \
} \
struct __useless_ignoreme

#define NUMERICAL_FUNCT_INTEGRAL(F, INTEGRATOR) \
namespace funct { \
  template<> struct Integral<F, no_var> { \
    typedef NumericalIntegral<INTEGRATOR, F> type; \
  }; \
} \
struct __useless_ignoreme

#endif
