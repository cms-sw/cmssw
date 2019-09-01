#ifndef PhysicsTools_Utilities_Factorize_h
#define PhysicsTools_Utilities_Factorize_h

#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"
#include "PhysicsTools/Utilities/interface/DecomposePower.h"
#include "PhysicsTools/Utilities/interface/DecomposeProduct.h"
#include <boost/type_traits.hpp>
#include <boost/integer/common_factor.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/if.hpp>

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {
  // find a common divider

  TEMPL(T1) struct Divides0 {
    static const bool value = false;
    typedef A arg;
    typedef NUM(1) type;
    GET(arg, num<1>());
  };

  TEMPL(T2) struct Divides : public Divides0<A> {};

  TEMPL(T2) struct Divides<PROD_S(A, B), void> : public Divides0<PROD_S(A, B)> {};

  template <TYPT1, bool par = Parametric<A>::value>
  struct ParametricDiv1 : public Divides<A, void> {};

  TEMPL(T1) struct ParametricDiv1<A, false> {
    static const bool value = true;
    typedef A arg;
    typedef arg type;
    GET(arg, _);
  };

  TEMPL(T1) struct Divides<A, A> : public ParametricDiv1<A> {};

  TEMPL(T2) struct Divides<PROD_S(A, B), PROD_S(A, B)> : public ParametricDiv1<PROD_S(A, B)> {};

  TEMPL(N1T1) struct Divides<POWER_S(A, NUM(n)), POWER_S(A, NUM(n))> : public ParametricDiv1<POWER_S(A, NUM(n))> {};

  template <TYPN2T1, bool par = Parametric<A>::value>
  struct ParametricDivN : public Divides<POWER(A, NUM(n)), void> {};

  TEMPL(N2T1) struct ParametricDivN<n, m, A, false> {
    static const bool value = true;
    typedef POWER(A, NUM(n)) arg;
    static const int p = ::boost::mpl::if_c<(n < m), ::boost::mpl::int_<n>, ::boost::mpl::int_<m> >::type::value;
    typedef POWER(A, NUM(p)) type;
    typedef DecomposePower<A, NUM(n)> Dec;
    GET(arg, pow(Dec::getBase(_), num<p>()));
  };

  TEMPL(N2T1) struct Divides<POWER_S(A, NUM(n)), POWER_S(A, NUM(m))> : public ParametricDivN<n, m, A> {};

  TEMPL(N1T1) struct Divides<A, POWER_S(A, NUM(n))> : public ParametricDivN<1, n, A> {};

  TEMPL(N1T1) struct Divides<POWER_S(A, NUM(n)), A> : public ParametricDivN<n, 1, A> {};

  namespace tmpl {

    template <int n, bool positive = (n >= 0)>
    struct abs {
      enum { value = n };
    };

    template <int n>
    struct abs<n, false> {
      enum { value = -n };
    };

  }  // namespace tmpl

  TEMPL(N2) struct Divides<NUM(n), NUM(m)> {
    enum { gcd = boost::integer::static_gcd<tmpl::abs<n>::value, tmpl::abs<m>::value>::value };
    static const bool value = (gcd != 1);
    typedef NUM(n) arg;
    typedef NUM(gcd) type;
    GET(arg, num<gcd>());
  };

  TEMPL(N1) struct Divides<NUM(n), NUM(n)> {
    static const bool value = true;
    typedef NUM(n) arg;
    typedef arg type;
    GET(arg, _);
  };

  TEMPL(T2) struct Divides<A, MINUS_S(B)> : public Divides<A, B> {};

  TEMPL(T3) struct Divides<PROD_S(A, B), MINUS_S(C)> : public Divides<PROD_S(A, B), C> {};

  TEMPL(T2) struct Divides<MINUS_S(A), B> {
    typedef Divides<A, B> Div;
    static const bool value = Div::value;
    typedef MINUS_S(A) arg;
    typedef typename Div::type type;
    GET(arg, Div::get(_._));
  };

  TEMPL(T2) struct Divides<MINUS_S(A), MINUS_S(B)> {
    typedef Divides<A, B> Div;
    static const bool value = Div::value;
    typedef MINUS_S(A) arg;
    typedef typename Div::type type;
    GET(arg, Div::get(_._));
  };

  TEMPL(T3) struct Divides<MINUS_S(A), PROD_S(B, C)> {
    typedef Divides<A, PROD_S(B, C)> Div;
    static const bool value = Div::value;
    typedef MINUS_S(A) arg;
    typedef typename Div::type type;
    GET(arg, Div::get(_._));
  };

  TEMPL(T3) struct Divides<A, PROD_S(B, C)> {
    typedef A arg;
    typedef Divides<arg, void> D0;
    typedef Divides<arg, B> D1;
    typedef Divides<arg, C> D2;
    typedef typename ::boost::mpl::if_<D1, D1, typename ::boost::mpl::if_<D2, D2, D0>::type>::type Div;
    static const bool value = Div::value;
    typedef typename Div::type type;
    GET(arg, Div::get(_));
  };

  TEMPL(T3) struct Divides<PROD_S(A, B), C> {
    typedef PROD_S(A, B) arg;
    typedef Divides<arg, void> D0;
    typedef Divides<A, C> D1;
    typedef Divides<B, C> D2;
    typedef typename ::boost::mpl::if_<D1, D1, typename ::boost::mpl::if_<D2, D2, D0>::type>::type Div;
    typedef typename Div::type type;
    static const bool value = Div::value;
    typedef DecomposeProduct<arg, typename Div::arg> D;
    GET(arg, Div::get(D::get(_)));
  };

  TEMPL(T4) struct Divides<PROD_S(A, B), PROD_S(C, D)> {
    typedef PROD_S(A, B) arg;
    typedef Divides<arg, void> D0;
    typedef Divides<arg, C> D1;
    typedef Divides<arg, D> D2;
    typedef typename ::boost::mpl::if_<D1, D1, typename ::boost::mpl::if_<D2, D2, D0>::type>::type Div;
    static const bool value = Div::value;
    typedef typename Div::type type;
    GET(arg, Div::get(_));
  };

  /*
    TEMPL(T4) struct Divides<RATIO_S(A, B), RATIO_S(C, D)> {
    typedef RATIO_S(A, B) arg;
    typedef Divides<B, D> Div;
    static const bool value = Div::value;
    DEF_TYPE(RATIO(NUM(1), typename Div::type))
    GET(arg, num(1) / Div::get(_))
    };
  */

  // general factorization algorithm
  template <TYPT2, bool factor = Divides<A, B>::value>
  struct FactorizeSum {
    typedef SUM_S(A, B) type;
    COMBINE(A, B, type(_1, _2));
  };

  TEMPL(T2) struct FactorizeSum<A, B, true> {
    typedef typename Divides<A, B>::type F;
    typedef PROD(F, SUM(RATIO(A, F), RATIO(B, F))) type;
    inline static type combine(const A& _1, const B& _2) {
      const F& f = Divides<A, B>::get(_1);
      return f * ((_1 / f) + (_2 / f));
    }
  };

  TEMPL(T3) struct Sum<PROD_S(A, B), C> : public FactorizeSum<PROD_S(A, B), C> {};

  TEMPL(T3) struct Sum<A, PROD_S(B, C)> : public FactorizeSum<A, PROD_S(B, C)> {};

  TEMPL(T3) struct Sum<MINUS_S(PROD_S(A, B)), C> : public FactorizeSum<MINUS_S(PROD_S(A, B)), C> {};

  TEMPL(T3) struct Sum<A, MINUS_S(PROD_S(B, C))> : public FactorizeSum<A, MINUS_S(PROD_S(B, C))> {};

  TEMPL(T4) struct Sum<PROD_S(A, B), PROD_S(C, D)> : public FactorizeSum<PROD_S(A, B), PROD_S(C, D)> {};

  TEMPL(T4)
  struct Sum<MINUS_S(PROD_S(A, B)), MINUS_S(PROD_S(C, D))>
      : public FactorizeSum<MINUS_S(PROD_S(A, B)), MINUS_S(PROD_S(C, D))> {};

  TEMPL(T4)
  struct Sum<PROD_S(A, B), MINUS_S(PROD_S(C, D))> : public FactorizeSum<PROD_S(A, B), MINUS_S(PROD_S(C, D))> {};

  TEMPL(T4)
  struct Sum<MINUS_S(PROD_S(A, B)), PROD_S(C, D)> : public FactorizeSum<MINUS_S(PROD_S(A, B)), PROD_S(C, D)> {};

  TEMPL(N1T2) struct Sum<PROD_S(A, B), NUM(n)> : public FactorizeSum<PROD_S(A, B), NUM(n)> {};

  TEMPL(N1T2) struct Sum<NUM(n), PROD_S(A, B)> : public FactorizeSum<NUM(n), PROD_S(A, B)> {};

  /*
    TEMPL(T4) struct Sum<SUM_S(A, B), PROD_S(C, D)> : 
    public FactorizeSum<SUM_S(A, B), PROD_S(C, D)> { };
    
    TEMPL(T4) struct Sum<SUM_S(A, B), MINUS_S(PROD_S(C, D))> : 
    public FactorizeSum<SUM_S(A, B), MINUS_S(PROD_S(C, D))> { };
    
    TEMPL(T4) struct Sum<PROD_S(A, B), SUM_S(C, D)> : 
    public FactorizeSum<PROD_S(A, B), SUM_S(C, D)> { };
  */

  /*
    template <TYPT4, bool factor = Divides<B, D>::value>
    struct FactorizeSumRatio {
    DEF_TYPE(SUM_S(RATIO_S(A, B), RATIO_S(C, D)))
    COMBINE(A, B, type(_1, _2))
    };
    
    TEMPL(T4) struct FactorizeSumRatio<A, B, C, D, true> {
    typedef typename Divides<B, D>::type F;
    DEF_TYPE(PROD(RATIO(NUM(1), F), 
    SUM(RATIO(PROD(A, F), B), 
    RATIO(PROD(C, F), D))))
    inline static type combine(const RATIO_S(A, B)& _1, 
    const RATIO_S(C, D)& _2) { 
    const F& f = Divides<B, D>::get(_1);       
    return (num(1) / f) * ((_1._1 * f) / _1._2 + (_2._1 * f) / _2._2);
    }
    };
    
    TEMPL(T4) struct Sum<RATIO_S(A, B), RATIO_S(C, D)> : 
    public FactorizeSum<RATIO_S(A, B), RATIO_S(C, D)> { };
  */
}  // namespace funct

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
