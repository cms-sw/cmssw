#ifndef PhysicsTools_Utilities_SimplifyRatio_h
#define PhysicsTools_Utilities_SimplifyRatio_h

#include "PhysicsTools/Utilities/interface/Ratio.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Minus.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/DecomposePower.h"
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

#include <boost/mpl/if.hpp>
#include <type_traits>

namespace funct {

  // 0 / a = 0
  RATIO_RULE(TYPT1, NUM(0), A, NUM(0) , num<0>());

  // a / 1 = a
  RATIO_RULE(TYPT1, A, NUM(1), A, _1);

  // ( a * b )/ 1 = a * b
  RATIO_RULE(TYPT2, PROD_S(A, B), NUM(1), PROD(A, B), _1);
  
  // a / ( -n ) = - ( a / n )
  template <int n, typename A, bool positive = (n>= 0)> 
  struct SimplifyNegativeRatio {
    typedef RATIO_S(A, NUM(n)) type;
    COMBINE(A, NUM(n), type(_1, _2));
  };

  TEMPL(N1T1)
  struct SimplifyNegativeRatio<n, A, false> {
    typedef MINUS(RATIO(A, NUM(-n))) type;
    COMBINE(A, NUM(n), - (_1 / num<-n>()));
  };

  TEMPL(N1T1) struct Ratio<A, NUM(n)> : 
    public SimplifyNegativeRatio<n, A> { };

  // ( -a ) / b = - ( a / b )
  RATIO_RULE(TYPT2, MINUS_S(A), B, MINUS(RATIO(A, B)), -(_1._ / _2));

  // ( -a ) / n = - ( a / n )
  RATIO_RULE(TYPN1T1, MINUS_S(A), NUM(n), MINUS(RATIO(A, NUM(n))), -(_1._ / _2));

  //TEMPL( N1T2 struct Ratio<PROD_S( A, B ), NUM( n )> : 
  //  public SimplifyNegativeRatio<n, PROD_S( A, B )> { };
  
  // n / ( m * a ) = (n/m) * a
  /* WRONG!!
  RATIO_RULE(TYPN2T1, NUM(n), PROD_S(NUM(m), A), \
	     PROD(FRACT(n, m), A), (fract<n, m>() * _2._2));
  */
  // ( a / b ) / c = a / ( b * c )
  RATIO_RULE(TYPT3, RATIO_S(A, B), C, \
	     RATIO(A, PROD(B, C)), _1._1 / (_1._2 * _2));
    
  // ( a / b ) / n = a / ( n * b )
  RATIO_RULE(TYPN1T2, RATIO_S(A, B), NUM(n), \
	     RATIO(A, PROD(NUM(n), B)), _1._1 / (_2 * _1._2));

  // ( a / b ) / ( c * d ) = a / ( b * c * d )
  RATIO_RULE(TYPT4, RATIO_S(A, B), PROD_S(C, D), \
	     RATIO(A, PROD(PROD(B, C), D)), _1._1 / (_1._2 * _2));

  // ( a * b ) / ( c / d ) = ( a * b * d ) / c
  RATIO_RULE(TYPT4, PROD_S(A, B), RATIO_S(C, D), \
	     RATIO(PROD(PROD(A, B), D), C), (_1 * _2._2) / _2._1);

  // ( n * a ) / ( m * b ) = ( n/m ) ( a / b )
  RATIO_RULE(TYPN2T2, PROD_S(NUM(n), A), PROD_S(NUM(m), B), \
	     PROD_S(FRACT(n, m), RATIO(A, B)), \
	     (PROD_S(FRACT(n, m), RATIO(A, B))((fract<n, m>()), (_1._2 / _2._2))));

  //  a / ( b / c ) = a * c / b
  RATIO_RULE(TYPT3, A, RATIO_S(B, C), \
	     RATIO(PROD(A, C), B), (_1 * _2._2) / _2._1);

  //  ( a + b ) / ( c / d ) = ( a + b ) * d / c
  RATIO_RULE(TYPT4, SUM_S(A, B), RATIO_S(C, D), \
	     RATIO(PROD(SUM(A, B), D), C), (_1 * _2._2) / _2._1);

  // ( a / b ) / ( c / d )= a * d / ( b * c )
  RATIO_RULE(TYPT4, RATIO_S(A, B), RATIO_S(C, D), \
	     RATIO(PROD(A, D), PROD(B, C)), \
	     (_1._1 * _2._2) / (_1._2 * _2._1));

  // ( a + b ) / ( b + a ) = 1
  template<TYPT2,
    bool parametric = (Parametric<A>::value == 1) || 
    (Parametric<B>::value == 1)>
  struct SimplifyRatioSum {
    typedef RATIO_S(SUM(A, B), SUM(B, A)) type;
    COMBINE(SUM(A, B), SUM(B, A), type(_1, _2));
  };
  
  TEMPL(T2) struct SimplifyRatioSum<A, B, false> {
    typedef NUM(1) type; 
    COMBINE(SUM(A, B), SUM(B, A), num<1>());
  };
  
  TEMPL(T2) struct Ratio<SUM_S(A, B), SUM_S(B, A)> : 
    public SimplifyRatioSum<A, B> { };
  
  // a^b / a^c => a^( b - c)
  template<TYPT3, bool parametric = (Parametric<A>::value == 1)>
  struct SimplifyPowerRatio {
    typedef POWER(A, B) arg1;
    typedef POWER(A, C) arg2;
    typedef RATIO_S(arg1, arg2) type;
    COMBINE(arg1, arg2, type(_1, _2));
  };
  
  TEMPL(T3) 
  struct SimplifyPowerRatio<A, B, C, false> {
    typedef POWER(A, B) arg1;
    typedef POWER(A, C) arg2;
    typedef POWER(A, DIFF(B, C)) type;
    inline static type combine(const arg1& _1, const arg2& _2) { 
      return pow(DecomposePower<A, B>::getBase(_1), 
		 (DecomposePower<A, B>::getExp(_1) - 
		  DecomposePower<A, C>::getExp(_2))); }
  };
  
  TEMPL(T3) struct Ratio<POWER_S(A, B), POWER_S(A, C)> : 
    public SimplifyPowerRatio<A, B, C> { };
  
  TEMPL(T2) struct Ratio<POWER_S(A, B), POWER_S(A, B)> :
    public SimplifyPowerRatio<A, B, B> { };
  
  TEMPL(T2) struct Ratio<A, POWER_S(A, B)> : 
    public SimplifyPowerRatio<A, NUM(1), B> { };
  
  TEMPL(N1T1) struct Ratio<A, POWER_S(A, NUM(n))> : 
    public SimplifyPowerRatio<A, NUM(1), NUM(n)> { };
  
  TEMPL(T2) struct Ratio<POWER_S(A, B), A> : 
    public SimplifyPowerRatio<A, B, NUM(1)>{ };
  
  TEMPL(N1T1) struct Ratio<POWER_S(A, NUM(n)), A> : 
    public SimplifyPowerRatio<A, NUM(n), NUM(1)> { };
  
  TEMPL(T1) struct Ratio<A, A> : 
    public SimplifyPowerRatio<A, NUM(1), NUM(1)> { };
  
  TEMPL(T2) struct Ratio<PROD_S(A, B), PROD_S(A, B)> : 
    public SimplifyPowerRatio<PROD_S(A, B), NUM(1), NUM(1)> { };
  
  TEMPL(N1T1) struct Ratio<PROD_S(NUM(n), A), PROD_S(NUM(n), A)> : 
    public SimplifyPowerRatio<PROD_S(NUM(n), A), NUM(1), NUM(1)> { };
  
  RATIO_RULE(TYPN1, NUM(n), NUM(n), NUM(1), num<1>());
    
  // simplify ( f * g ) / h
  // try ( f / h ) * g and ( g / h ) * f, otherwise leave ( f * g ) / h
  
  template <typename Prod, bool simplify = Prod::value> struct AuxProductRatio {
    typedef PROD(typename Prod::AB, typename Prod::C) type;
    inline static type combine(const typename Prod::A& a, 
			       const typename Prod::B& b, 
			       const typename Prod::C& c) { return (a / b) * c; }
  };
  
  template<typename Prod>  struct AuxProductRatio<Prod, false> {
    typedef RATIO_S(typename Prod::AB, typename Prod::C) type;
    inline static type combine(const typename Prod::A& a,
			       const typename Prod::B& b, 
			       const typename Prod::C& c) { return type(a * b, c); }
  };
  
  template<typename F, typename G, typename H>
  struct RatioP1 {
    struct prod0 { 
      typedef F A; typedef G B; typedef H C;
      typedef PROD_S(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return g; }
      inline static const C& c(const F& f, const G& g, const H& h) { return h; }
      enum { value = false };
    };
    struct prod1 { 
      typedef F A; typedef H B; typedef G C;
      typedef RATIO_S(A, B) base;
      typedef RATIO(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return g; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    struct prod2 { 
      typedef G A; typedef H B; typedef F C;
      typedef RATIO_S(A, B) base;
      typedef RATIO(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return g; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return f; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    
    typedef typename 
    ::boost::mpl::if_<prod1, 
		      prod1,
		      typename ::boost::mpl::if_<prod2,
						 prod2,
						 prod0 
						 >::type
		      >::type prod;
    typedef typename AuxProductRatio<prod>::type type;
    inline static type combine(const PROD_S(F, G)& fg, const H& h) {
      const F& f = fg._1;
      const G& g = fg._2;
      const typename prod::A & a = prod::a(f, g, h);
      const typename prod::B & b = prod::b(f, g, h);
      const typename prod::C & c = prod::c(f, g, h);
      return AuxProductRatio<prod>::combine(a, b, c); 
    }
  };
  
  // simplify c / ( a * b )
  // try ( c / a ) / b and ( c / b ) / a, otherwise leave c / ( a * b )
  
  template <typename Prod, bool simplify = Prod::value> 
  struct AuxProductRatio2 {
    typedef RATIO(typename Prod::AB, typename Prod::C) type;
    inline static type combine(const typename Prod::A& a, 
			       const typename Prod::B& b, 
			       const typename Prod::C& c) { return (b / a) / c; }
  };
  
  template<typename Prod>  
  struct AuxProductRatio2<Prod, false> {
    typedef RATIO_S(typename Prod::C, typename Prod::AB) type;
    inline static type combine(const typename Prod::A& a,
			       const typename Prod::B& b, 
			       const typename Prod::C& c) { return type(c, a * b); }
  };
  
  template<typename F, typename G, typename H>
  struct RatioP2 {
    struct prod0 { 
      typedef F A; typedef G B; typedef H C;
      typedef PROD_S(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return g; }
      inline static const C& c(const F& f, const G& g, const H& h) { return h; }
      enum { value = false };
    };
    struct prod1 { 
      typedef F A; typedef H B; typedef G C;
      typedef RATIO_S(B, A) base;
      typedef RATIO(B, A) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return g; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    struct prod2 { 
      typedef G A; typedef H B; typedef F C;
      typedef RATIO_S(B, A) base;
      typedef RATIO(B, A) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return g; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return f; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    
    typedef typename 
    ::boost::mpl::if_<prod1, 
		      prod1,
		      typename ::boost::mpl::if_<prod2,
						 prod2,
						 prod0 
						 >::type
		     >::type prod;
    typedef typename AuxProductRatio2<prod>::type type;
    inline static type combine(const H& h, const PROD_S(F, G)& fg) {
      const F& f = fg._1;
      const G& g = fg._2;
      const typename prod::A & a = prod::a(f, g, h);
      const typename prod::B & b = prod::b(f, g, h);
      const typename prod::C & c = prod::c(f, g, h);
      return AuxProductRatio2<prod>::combine(a, b, c); 
    }
  };
  
  TEMPL(T3) struct Ratio<PROD_S(A, B), C> :
    public RatioP1<A, B, C> { };
  
  TEMPL(N1T2) struct Ratio<PROD_S(A, B), NUM(n)> :
    public RatioP1<A, B, NUM(n)> { };
  
  TEMPL(T3) struct Ratio<C, PROD_S(A, B)> :
    public RatioP2<A, B, C> { };
  
  TEMPL(T4) struct Ratio<PROD_S(C, D), PROD_S(A, B)> :
    public RatioP2<A, B, PROD_S(C, D)> { };
  
  // simplify ( a + b ) / c trying to simplify ( a / c ) and ( b / c ) 
  template <TYPT3, bool simplify = false> struct AuxSumRatio {
    typedef RATIO_S(SUM_S(A, B), C) type;
    COMBINE(SUM_S(A, B), C, type(_1, _2));
  };
  
  TEMPL(T3) struct AuxSumRatio<A, B, C, true> {
    typedef SUM(RATIO(A, C), RATIO(B, C)) type;
    COMBINE(SUM_S(A, B), C, (_1._1 / _2) + (_1._2 / _2));
  };
  
  TEMPL(T3) struct RatioSimpl {
    struct ratio1 { 
      typedef RATIO_S(A, C) base;
      typedef RATIO(A, C) type;
      enum { value = not ::std::is_same<type, base>::value };
    };
    struct ratio2 { 
      typedef RATIO_S(B, C) base;
      typedef RATIO(B, C) type;
      enum { value = not ::std::is_same<type, base>::value };
    };
    typedef AuxSumRatio<A, B, C, ratio1::value or ratio2::value> aux;
    typedef typename aux::type type;
    COMBINE(SUM_S(A, B), C, aux::combine(_1, _2));
  };

  TEMPL(T3) struct Ratio<SUM_S(A, B), C> : 
    public RatioSimpl<A, B, C> { };
  
  TEMPL(T4) struct Ratio<SUM_S(A, B), PROD_S(C, D)> : 
    public RatioSimpl<A, B, PROD_S(C, D)> { };
  
  TEMPL(N1T2) struct Ratio<SUM_S(A, B), NUM(n)> : 
    public RatioSimpl<A, B, NUM(n)> { };

}

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
