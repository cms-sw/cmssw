#ifndef PhysicsTools_Utilities_Primitive_h
#define PhysicsTools_Utilities_Primitive_h
#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Derivative.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Identity.h"
#include <type_traits>

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  struct no_var;

  struct UndefinedIntegral {};

  template <typename X, typename F, bool independent = Independent<X, F>::value>
  struct ConstPrimitive {
    typedef UndefinedIntegral type;
    inline static type get(const F& f) { return type(); }
  };

  //  /
  //  | c dx = c * x
  //  /
  template <typename X, typename F>
  struct ConstPrimitive<X, F, true> {
    typedef PROD(F, X) type;
    inline static type get(const F& f) { return f * X(); }
  };

  template <typename F, typename X = no_var>
  struct Primitive : public ConstPrimitive<X, F> {};

  template <typename F>
  struct Primitive<F> {};

  template <typename X, typename F>
  typename Primitive<F, X>::type primitive(const F& f) {
    return Primitive<F, X>::get(f);
  }

  template <typename F>
  typename Primitive<F>::type primitive(const F& f) {
    return Primitive<F>::get(f);
  }

  //  /
  //  | f ^ g dx : UNDEFINED
  //  /
  PRIMIT_RULE(TYPXT2, POWER_S(A, B), UndefinedIntegral, type());

  //  /
  //  | x dx = x ^ 2 / 2
  //  /
  PRIMIT_RULE(TYPX, X, RATIO(POWER(X, NUM(2)), NUM(2)), pow(_, num<2>()) / num<2>());

  //  /
  //  | x ^ n dx = x ^ (n + 1) / (n + 1)
  //  /
  PRIMIT_RULE(TYPXN1,
              POWER_S(X, NUM(n)),
              RATIO(POWER(X, NUM(n + 1)), NUM(n + 1)),
              pow(_._1, num<n + 1>()) / num<n + 1>());

  //  /
  //  | 1 / x ^ n dx = (- 1) / ((n - 1)  x ^ (n - 1))
  //  /
  PRIMIT_RULE(TYPXN1,
              RATIO_S(NUM(1), POWER_S(X, NUM(n))),
              RATIO(NUM(-1), PROD(NUM(n - 1), POWER(X, NUM(n - 1)))),
              num<-1>() / (num<n - 1>() * pow(_._2._1, num<n - 1>())));

  PRIMIT_RULE(TYPXN1,
              POWER_S(RATIO_S(NUM(1), X), NUM(n)),
              RATIO(NUM(-1), PROD(NUM(n - 1), POWER(X, NUM(n - 1)))),
              num<-1>() / (num<n - 1>() * pow(_._1._2, num<n - 1>())));

  //  /
  //  | x ^ n/m dx = m / (n + m) (x)^ (n + m / m)
  //  /
  PRIMIT_RULE(TYPXN2,
              POWER_S(X, FRACT_S(n, m)),
              PROD(FRACT(m, n + m), POWER(X, FRACT(n + m, m))),
              (fract<m, n + m>() * pow(_._1, fract<n + m, m>())));

  //  /
  //  | sqrt(x) dx = 2/3 (x)^ 3/2
  //  /
  PRIMIT_RULE(TYPX, SQRT_S(X), PRIMIT(X, POWER_S(X, FRACT_S(1, 2))), (fract<2, 3>() * pow(_._, fract<3, 2>())));

  //  /
  //  | exp(x) dx = exp(x)
  //  /
  PRIMIT_RULE(TYPX, EXP_S(X), EXP(X), _);

  //  /
  //  | log(x) dx = x(log(x) - 1)
  //  /
  PRIMIT_RULE(TYPX, LOG_S(X), PROD(X, DIFF(LOG(X), NUM(1))), _._*(_ - num<1>()));

  //  /
  //  | sgn(x) dx = abs(x)
  //  /
  PRIMIT_RULE(TYPX, SGN_S(X), ABS(X), abs(_._));

  //  /
  //  | sin(x) dx = - cos(x)
  //  /
  PRIMIT_RULE(TYPX, SIN_S(X), MINUS(COS(X)), -cos(_._));

  //  /
  //  | cos(x) dx = sin(x)
  //  /
  PRIMIT_RULE(TYPX, COS_S(X), SIN(X), sin(_._));

  //  /
  //  | tan(x) dx = - log(abs(cos(x)))
  //  /
  PRIMIT_RULE(TYPX, TAN_S(X), MINUS(LOG(ABS(COS(X)))), -log(abs(cos(_._))));

  //  /
  //  | 1 / x dx = log(abs(x))
  //  /
  PRIMIT_RULE(TYPX, RATIO_S(NUM(1), X), LOG(ABS(X)), log(abs(_._2)));

  PRIMIT_RULE(TYPX, POWER_S(X, NUM(-1)), LOG(ABS(X)), log(abs(_._1)));

  //  /
  //  | 1 / cos(x)^2 dx = tan(x)
  //  /
  PRIMIT_RULE(TYPX, RATIO_S(NUM(1), POWER_S(COS_S(X), NUM(2))), TAN(X), tan(_._2._1._));

  //  /
  //  | 1 / sin(x)^2 dx = - 1 / tan(x)
  //  /
  PRIMIT_RULE(TYPX, RATIO_S(NUM(1), POWER_S(SIN_S(X), NUM(2))), RATIO(NUM(-1), TAN(X)), num<-1>() / tan(_._2._1._));

  // composite primitives

  //  /                    /           /
  //  | (f(x) + g(x)) dx = | f(x) dx + | g(x) dx
  //  /                    /           /
  PRIMIT_RULE(TYPXT2, SUM_S(A, B), SUM(PRIMIT(X, A), PRIMIT(X, B)), primitive<X>(_._1) + primitive<X>(_._2));

  //  /                 /
  //  | (- f(x)) dx = - | f(x) dx
  //  /                 /
  PRIMIT_RULE(TYPXT1, MINUS_S(A), MINUS(PRIMIT(X, A)), -primitive<X>(_._));

  //  /
  //  | f * g dx : defined only for f or g indep. of x or part. int.
  //  /

  template <TYPXT2,
            bool bint = not ::std::is_same<PRIMIT(X, B), UndefinedIntegral>::value,
            bool aint = not ::std::is_same<PRIMIT(X, A), UndefinedIntegral>::value>
  struct PartIntegral {
    typedef UndefinedIntegral type;
    GET(PROD_S(A, B), type());
  };

  TEMPL(XT2) struct PartIntegral<X, A, B, true, false> {
    typedef PRIMIT(X, B) B1;
    typedef DERIV(X, A) A1;
    typedef PRIMIT(X, PROD(A1, B1)) AB1;
    typedef DIFF(PROD(A, B1), PRIMIT(X, PROD(A1, B1))) type;
    inline static type get(const PROD_S(A, B) & _) {
      const A& a = _._1;
      B1 b = primitive<X>(_._2);
      return a * b - primitive<X>(derivative<X>(a) * b);
    }
  };

  TEMPL(XT2) struct PartIntegral<X, B, A, false, true> {
    typedef PRIMIT(X, B) B1;
    typedef DERIV(X, A) A1;
    typedef PRIMIT(X, PROD(A1, B1)) AB1;
    typedef DIFF(PROD(A, B1), PRIMIT(X, PROD(A1, B1))) type;
    inline static type get(const PROD_S(B, A) & _) {
      const A& a = _._2;
      B1 b = primitive<X>(_._1);
      return a * b - primitive<X>(derivative<X>(a) * b);
    }
  };

  TEMPL(XT2) struct PartIntegral<X, A, B, true, true> : public PartIntegral<X, A, B, true, false> {};

  template <TYPXT2, bool indepf = Independent<X, A>::value, bool indepg = Independent<X, B>::value>
  struct ProductPrimitive : public PartIntegral<X, A, B> {};

  TEMPL(XT2) struct ProductPrimitive<X, A, B, true, false> {
    typedef PROD(A, PRIMIT(X, B)) type;
    GET(PROD_S(A, B), _._1* primitive<X>(_._2));
  };

  TEMPL(XT2) struct ProductPrimitive<X, A, B, false, true> {
    typedef PROD(B, PRIMIT(X, A)) type;
    GET(PROD_S(A, B), _._2* primitive<X>(_._1));
  };

  TEMPL(XT2) struct ProductPrimitive<X, A, B, true, true> {
    typedef PROD(PROD(A, B), X) type;
    GET(PROD_S(A, B), _* X());
  };

  TEMPL(XT2) struct Primitive<PROD_S(A, B), X> : public ProductPrimitive<X, A, B> {};

  //  /
  //  | f / g dx : defined only for f or g indep. of x; try part. int.
  //  /

  template <TYPXT2,
            bool bint = not ::std::is_same<PRIMIT(X, RATIO(NUM(1), B)), UndefinedIntegral>::value,
            bool aint = not ::std::is_same<PRIMIT(X, A), UndefinedIntegral>::value>
  struct PartIntegral2 {
    typedef UndefinedIntegral type;
    GET(RATIO_S(A, B), type());
  };

  TEMPL(XT2) struct PartIntegral2<X, A, B, true, false> {
    typedef PRIMIT(X, RATIO(NUM(1), B)) B1;
    typedef DERIV(X, A) A1;
    typedef PRIMIT(X, PROD(A1, B1)) AB1;
    typedef DIFF(PROD(A, B1), AB1) type;
    inline static type get(const RATIO_S(A, B) & _) {
      const A& a = _._1;
      B1 b = primitive<X>(num<1>() / _._2);
      return a * b - primitive<X>(derivative<X>(a) * b);
    }
  };

  TEMPL(XT2) struct PartIntegral2<X, B, A, false, true> {
    typedef PRIMIT(X, RATIO(NUM(1), B)) B1;
    typedef DERIV(X, A) A1;
    typedef PRIMIT(X, PROD(A1, B1)) AB1;
    typedef DIFF(PROD(A, B1), AB1) type;
    inline static type get(const RATIO_S(B, A) & _) {
      const A& a = _._1;
      B1 b = primitive<X>(num<1>() / _._2);
      return a * b - primitive<X>(derivative<X>(a) * b);
    }
  };

  // should be improved: try both...
  TEMPL(XT2) struct PartIntegral2<X, A, B, true, true> : public PartIntegral2<X, A, B, true, false> {};

  template <TYPXT2, bool indepa = Independent<X, A>::value, bool indepb = Independent<X, B>::value>
  struct RatioPrimitive : public PartIntegral2<X, A, B> {};

  TEMPL(XT2) struct RatioPrimitive<X, A, B, true, false> {
    typedef PROD(A, PRIMIT(X, RATIO(NUM(1), B))) type;
    GET(RATIO_S(A, B), _._1* primitive<X>(num<1> / _._2));
  };

  TEMPL(XT2) struct RatioPrimitive<X, A, B, false, true> {
    typedef RATIO(PRIMIT(X, A), B) type;
    GET(RATIO_S(A, B), primitive<X>(_._1) / _._2);
  };

  TEMPL(XT2) struct RatioPrimitive<X, A, B, true, true> {
    typedef RATIO(RATIO(A, B), X) type;
    GET(RATIO_S(A, B), _* X());
  };

  TEMPL(XT2) struct Primitive<RATIO_S(A, B), X> : public RatioPrimitive<X, A, B> {};

  // Function integrals

  //  /
  //  | c dx = c * x
  //  /
  template <>
  struct Primitive<Parameter> {
    typedef Product<Parameter, Identity>::type type;
    inline static type get(const Parameter& p) { return p * Identity(); }
  };

}  // namespace funct

#define DECLARE_PRIMITIVE(X, F, P)                             \
  namespace funct {                                            \
    template <typename X>                                      \
    struct Primitive<F, X> {                                   \
      typedef P type;                                          \
      inline static type get(const F& _) { return type(_._); } \
    };                                                         \
  }                                                            \
  struct __useless_ignoreme

#define DECLARE_FUNCT_PRIMITIVE(F, P)                       \
  namespace funct {                                         \
    template <>                                             \
    struct Primitive<F> {                                   \
      typedef P type;                                       \
      inline static type get(const F& _) { return type(); } \
    };                                                      \
  }                                                         \
  struct __useless_ignoreme

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
