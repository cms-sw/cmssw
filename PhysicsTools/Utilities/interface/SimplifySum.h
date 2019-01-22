#ifndef PhysicsTools_Utilities_SimplifySum_h
#define PhysicsTools_Utilities_SimplifySum_h

#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Difference.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/DecomposeProduct.h"
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"
#include <type_traits>
#include <boost/mpl/if.hpp>

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  // A + ( B + C ) => ( A + B ) + C
  SUM_RULE(TYPT3, A, SUM_S(B, C),
	   SUM(SUM(A, B), C), (_1 + _2._1) + _2._2);

  // ( A + B ) + ( C + D ) => ( ( A + B ) + C ) + D
  SUM_RULE(TYPT4, SUM_S(A, B), SUM_S(C, D),
	   SUM(SUM(SUM(A, B), C), D), (_1 + _2._1) + _2._2);

  // n + A = A + n
  SUM_RULE(TYPN1T1, NUM(n), A, SUM(A, NUM(n)), _2 + _1);

  // n + ( A + B )= ( A + B ) + n
  SUM_RULE(TYPN1T2, NUM(n), SUM_S(A, B), SUM(SUM_S(A, B), NUM(n)), _2 + _1);

  // A + 0 = A
  SUM_RULE(TYPT1, A, NUM(0), A, _1);

  // 0 + 0 = 0
  SUM_RULE(TYP0, NUM(0), NUM(0), NUM(0), num<0>());

  // ( A * B ) + 0 = ( A * B )
  SUM_RULE(TYPT2, PROD_S(A, B), NUM(0), PROD_S(A, B), _1);

  // 0 + ( A * B ) = ( A * B )
  SUM_RULE(TYPT2, NUM(0), PROD_S(A, B), PROD_S(A, B), _2);

  // 0 - ( A * B ) = - ( A * B )
  SUM_RULE(TYPT2, NUM(0), MINUS_S(PROD_S(A, B)), MINUS_S(PROD_S(A, B)), _2);

  // ( A + B ) + 0 = ( A + B )
  SUM_RULE(TYPT2, SUM_S(A, B), NUM(0), SUM_S(A, B), _1);

  // 0 + ( A + B ) = ( A + B )
  SUM_RULE(TYPT2, NUM(0), SUM_S(A, B), SUM_S(A, B), _2);

  // A - ( -B ) =  A + B
  DIFF_RULE(TYPT2, A, MINUS_S(B), SUM(A, B), _1 + _2._);

  // n * A + m * A => ( n + m ) * A
  template<TYPN2T1, bool parametric = Parametric<A>::value == 1>
    struct ParametricSimplifiedSum {
      typedef PROD(NUM(n), A) arg1;
      typedef PROD(NUM(m), A) arg2;
      typedef SUM_S(arg1, arg2) type;
      COMBINE(arg1, arg2, type(_1, _2));
    };

  TEMPL(N2T1) 
  struct ParametricSimplifiedSum<n, m, A, false> {
    typedef PROD(NUM(n + m), A) type;
    typedef DecomposeProduct<PROD(NUM(n), A), A> Dec;
    COMBINE(PROD(NUM(n), A), PROD(NUM(m), A),
	    num<n + m>() * Dec::get(_1));
  };

  TEMPL(T1) 
  struct ParametricSimplifiedSum<1, 1, A, true> {
    typedef SumStruct<A, A> type;
    COMBINE(A, A, type(_1, _2));
  };

  TEMPL(T1) 
  struct ParametricSimplifiedSum<1, 1, A, false> {
    typedef PROD(NUM(2), A) type;
    COMBINE( A, A, num<2>() * _1 );
  };

  TEMPL(N2T1) 
  struct Sum<PROD_S(NUM(n), A), PROD_S(NUM(m), A) > : 
    public ParametricSimplifiedSum<n, m, A> { };

  TEMPL(N1T1) 
  struct Sum<A, PROD_S(NUM(n), A) > : 
    public ParametricSimplifiedSum<1, n, A> { };

  TEMPL(N1T1) 
  struct Sum<PROD_S(NUM(n), A) , A> : 
    public ParametricSimplifiedSum<n, 1, A> { };

  TEMPL(T1) 
  struct Sum<A, A> : 
    public ParametricSimplifiedSum<1, 1, A> { };

  TEMPL(T1) 
  struct Sum<MINUS_S(A), MINUS_S(A) > : 
    public ParametricSimplifiedSum<1, 1, MINUS_S(A) > { };

  TEMPL(T2) 
  struct Sum< MINUS_S(PROD_S(A, B)), 
    MINUS_S(PROD_S(A, B)) > : 
    public ParametricSimplifiedSum< 1, 1, MINUS_S(PROD_S(A, B)) > { };

  TEMPL(N1) 
  struct Sum< NUM(n), NUM(n) > : 
    public ParametricSimplifiedSum< 1, 1, NUM(n) > { };

  TEMPL(T2) 
  struct Sum< PROD_S(A, B), PROD_S(A, B) > : 
    public ParametricSimplifiedSum< 1, 1, PROD_S(A, B) > { };

  TEMPL(N1T1) 
  struct Sum< PROD_S(NUM(n), A), 
    PROD_S(NUM(n), A) > : 
    public ParametricSimplifiedSum< 1, 1, PROD_S(NUM(n), A) > { };

  // simplify f + g + h regardless of the order 
  template <typename Prod, bool simplify = Prod::value> 
  struct AuxSum {
    typedef SUM(typename Prod::AB, typename Prod::C) type;
    COMBINE(typename Prod::AB, typename Prod::C, _1 + _2);
  };
  
  template<typename Prod>  
  struct AuxSum<Prod, false> {
    typedef SUM_S(typename Prod::AB, typename Prod::C) type;
    COMBINE(typename Prod::AB, typename Prod::C, type(_1, _2));
  };

  template<typename F, typename G, typename H>
  struct SimplSumOrd {
    struct prod0 { 
      typedef F A; typedef G B; typedef H C;
      typedef SUM_S(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return g; }
      inline static const C& c(const F& f, const G& g, const H& h) { return h; }
      enum { value = false };
    };
    struct prod1 { 
      typedef F A; typedef H B; typedef G C;
      typedef SUM_S(A, B) base;
      typedef SUM(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return g; }
      enum { value = not ::std::is_same< AB, base >::value };
    };
    struct prod2 { 
      typedef G A; typedef H B; typedef F C;
      typedef SUM_S(A, B) base;
      typedef SUM(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return g; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return f; }
      enum { value = not ::std::is_same< AB, base >::value };
    };
    
    typedef typename 
    ::boost::mpl::if_ <prod1, 
		       prod1,
		       typename ::boost::mpl::if_ <prod2,
						   prod2,
						   prod0 
						   >::type
		       >::type prod;
    typedef typename AuxSum< prod >::type type;
    inline static type combine(const SUM_S(F, G)& fg, const H& h) {
      const F& f = fg._1;
      const G& g = fg._2;
      const typename prod::A & a = prod::a(f, g, h);
      const typename prod::B & b = prod::b(f, g, h);
      const typename prod::C & c = prod::c(f, g, h);
      return AuxSum< prod >::combine(a + b, c); 
    }
  };
  
  TEMPL(T3) 
  struct Sum<SUM_S(A, B), C> : 
    public SimplSumOrd<A, B, C> { };
  
  TEMPL(T4) 
  struct Sum< SUM_S(A, B), PROD_S(C, D) > : 
    public SimplSumOrd< A, B, PROD_S(C, D) > { };

}

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
