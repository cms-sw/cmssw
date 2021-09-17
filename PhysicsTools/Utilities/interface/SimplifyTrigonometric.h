#ifndef PhysicsTools_Utilities_SimplifyTrigonometric_h
#define PhysicsTools_Utilities_SimplifyTrigonometric_h

#include "PhysicsTools/Utilities/interface/Sin.h"
#include "PhysicsTools/Utilities/interface/Cos.h"
#include "PhysicsTools/Utilities/interface/Tan.h"
#include "PhysicsTools/Utilities/interface/Sin2Cos2.h"
#include "PhysicsTools/Utilities/interface/Minus.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Ratio.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"
#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {
  // sin(-a) = - sin(a)
  SIN_RULE(TYPT1, MINUS_S(A), MINUS(SIN(A)), -sin(_._));

  // cos(-a) = cos(a)
  COS_RULE(TYPT1, MINUS_S(A), COS(A), cos(_._));

  // tan(-a) = - tan(a)
  TAN_RULE(TYPT1, MINUS_S(A), MINUS(TAN(A)), -tan(_._));

  // sin(x) * x = x * sin(x)
  PROD_RULE(TYPT1, SIN_S(A), A, PROD(A, SIN(A)), _2* _1);

  // cos(x) * x = x * cos(x)
  PROD_RULE(TYPT1, COS_S(A), A, PROD(A, COS(A)), _2* _1);

  // tan(x) * x = x * tan(x)
  PROD_RULE(TYPT1, TAN_S(A), A, PROD(A, TAN(A)), _2* _1);

  // sin(a) / cos(a) = tan(a)
  template <TYPT1, bool parametric = Parametric<A>::value>
  struct SimplifySCRatio {
    typedef RATIO_S(SIN(A), COS(A)) type;
    COMBINE(SIN_S(A), COS_S(A), _1 / _2);
  };

  TEMPL(T1) struct SimplifySCRatio<A, false> {
    typedef TAN_S(A) type;
    COMBINE(SIN_S(A), COS_S(A), type(_1._));
  };

  TEMPL(T1) struct Ratio<SIN_S(A), COS_S(A)> : public SimplifySCRatio<A> {};

  // sin(a) / tan(a) = cos(a)
  template <TYPT1, bool parametric = Parametric<A>::value>
  struct SimplifySTRatio {
    typedef RATIO_S(SIN(A), TAN(A)) type;
    COMBINE(SIN_S(A), TAN_S(A), _1 / _2);
  };

  TEMPL(T1) struct SimplifySTRatio<A, false> {
    typedef COS_S(A) type;
    COMBINE(SIN_S(A), TAN_S(A), type(_1._));
  };

  TEMPL(T1) struct Ratio<SIN_S(A), TAN_S(A)> : public SimplifySTRatio<A> {};

  // cos(a) * tan(a) = sin(a)
  template <TYPT1, bool parametric = Parametric<A>::value>
  struct SimplifySTProduct {
    typedef PROD(COS(A), TAN(A)) type;
    COMBINE(COS_S(A), TAN_S(A), _1* _2);
  };

  TEMPL(T1) struct SimplifySTProduct<A, false> {
    typedef SIN(A) type;
    COMBINE(COS_S(A), TAN_S(A), sin(_1._));
  };

  TEMPL(T1) struct Product<COS_S(A), TAN_S(A)> : public SimplifySTProduct<A> {};

  // cos(a) * sin(a) => sin(a) * cos(a)
  TEMPL(T1) struct Product<COS_S(A), SIN_S(A)> {
    typedef PROD(SIN(A), COS(A)) type;
    COMBINE(COS_S(A), SIN_S(A), _2* _1);
  };

  // cos(a)^b * tan(a)^b = sin(a)^b
  template <TYPT2, bool parametric = Parametric<A>::value || Parametric<B>::value>
  struct SimplifySTnProduct {
    typedef PROD(POWER(COS(A), B), POWER(TAN(A), B)) type;
    COMBINE(POWER_S(COS_S(A), B), POWER_S(TAN_S(A), B), _1* _2);
  };

  TEMPL(T2) struct SimplifySTnProduct<A, B, false> {
    typedef POWER(SIN(A), B) type;
    COMBINE(POWER_S(COS_S(A), B), POWER_S(TAN_S(A), B), pow(sin(_1._1._), _1._2));
  };

  TEMPL(T2) struct Product<POWER_S(COS_S(A), B), POWER_S(TAN_S(A), B)> : public SimplifySTnProduct<A, B> {};

  TEMPL(N1T1)
  struct Product<POWER_S(COS_S(A), NUM(n)), POWER_S(TAN_S(A), NUM(n))> : public SimplifySTnProduct<A, NUM(n)> {};

  // n cos(a)^2 + m sin(a)^2 = min(n, m) +
  //        (n - min(n, m)) cos(a)^2 + (m - min(n, m)) sin(a)^2
  template <TYPN2T1, bool parametric = Parametric<A>::value>
  struct SimpifyS2C2Sum {
    typedef SUM(PROD(NUM(n), SIN2(A)), PROD(NUM(m), COS2(A))) type;
    COMBINE(PROD(NUM(n), SIN2(A)), PROD(NUM(m), COS2(A)), _1 + _2);
  };

  TEMPL(N2T1) struct SimpifyS2C2Sum<n, m, A, false> {
    static constexpr int p = n < m ? n : m;
    typedef SUM(SUM(PROD(NUM(n - p), SIN2(A)), PROD(NUM(m - p), COS2(A))), NUM(p)) type;
    COMBINE(PROD(NUM(n), SIN2(A)), PROD(NUM(m), COS2(A)), (num<n - p>() * _1._2 + num<m - p>() * _2._2) + num<p>());
  };

  TEMPL(T1) struct Sum<POWER_S(SIN_S(A), NUM(2)), POWER_S(COS_S(A), NUM(2))> : public SimpifyS2C2Sum<1, 1, A> {};

  TEMPL(T1) struct Sum<POWER_S(COS_S(A), NUM(2)), POWER_S(SIN_S(A), NUM(2))> {
    typedef SUM(SIN2(A), COS2(A)) type;
    inline static type combine(const COS2(A) & _1, const SIN2(A) & _2) {
      return Sum<SIN2(A), COS2(A)>::combine(_2, _1);
    }
  };

  TEMPL(N2T1)
  struct Sum<PROD_S(NUM(n), POWER_S(SIN_S(A), NUM(2))), PROD_S(NUM(m), POWER_S(COS_S(A), NUM(2)))>
      : public SimpifyS2C2Sum<n, m, A> {};

  TEMPL(N2T1) struct Sum<PROD_S(NUM(m), POWER_S(COS_S(A), NUM(2))), PROD_S(NUM(n), POWER_S(SIN_S(A), NUM(2)))> {
    typedef SUM(PROD(NUM(n), SIN2(A)), PROD(NUM(m), COS2(A))) type;
    inline static type combine(const PROD(NUM(m), COS2(A)) & _1, const PROD(NUM(n), SIN2(A)) & _2) {
      return Sum<PROD(NUM(n), SIN2(A)), PROD(NUM(m), COS2(A))>::combine(_2, _1);
    }
  };

}  // namespace funct

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
