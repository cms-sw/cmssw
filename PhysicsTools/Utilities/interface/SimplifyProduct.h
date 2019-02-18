#ifndef PhysicsTools_Utilities_SimplifyProduct_h
#define PhysicsTools_Utilities_SimplifyProduct_h

#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/DecomposePower.h"
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"
#include <boost/mpl/if.hpp>
#include <type_traits>

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  // a * ( b * c ) = ( a * b ) * c
  PROD_RULE(TYPT3, A, PROD_S(B, C),
	    PROD(PROD(A, B), C), (_1 * _2._1) * _2._2);

  // 0 * a = 0
  PROD_RULE(TYPT1, NUM(0), A, NUM(0), num<0>());
  
  // 0 * n = 0
  PROD_RULE(TYPN1, NUM(0), NUM(n), NUM(0), num<0>());

  // 0 * ( a * b ) => 0
  PROD_RULE(TYPT2, NUM(0), PROD_S(A, B), NUM(0), num<0>());
  
  // 1 * a = a
  PROD_RULE(TYPT1, NUM(1), A, A, _2);

  // avoid template ambiguities
  // 1 * n = n
  PROD_RULE(TYPN1, NUM(1), NUM(n), NUM(n), _2);

  // 1 * (n/m) = (n/m)
  PROD_RULE(TYPN2, NUM(1), FRACT_S(n, m), FRACT_S(n, m), _2);
    
  // 1 * 1 = 1
  PROD_RULE(TYP0, NUM(1), NUM(1), NUM(1), num<1>());

  // ( - 1 ) * a = - a
  PROD_RULE(TYPT1, NUM(-1), A, MINUS_S(A), -_2);

  // ( - 1 ) * n = -n
  PROD_RULE(TYPN1, NUM(-1), NUM(n), NUM(-n), num<-n>());

  // 1 * ( a * b ) => ( a * b )
  PROD_RULE(TYPT2, NUM(1), PROD_S(A, B), PROD_S(A, B), _2);

  // a * ( -b ) =  - ( a * b )
  PROD_RULE(TYPT2, A, MINUS_S(B), MINUS(PROD(A, B)), -(_1 * _2._));

  // n * ( -a ) =  - ( n * a )
  PROD_RULE(TYPN1T1, NUM(n), MINUS_S(A), MINUS(PROD(NUM(n), A)), -(_1 * _2._ )); 

  // ( a * b ) * ( -c )=  - ( ( a * b ) * c )
  PROD_RULE(TYPT3, PROD_S(A, B), MINUS_S(C), MINUS(PROD(PROD(A, B), C)), -(_1 * _2._));

  // 1 * ( -a ) = -a
  PROD_RULE(TYPT1, NUM(1), MINUS_S(A), MINUS(A), _2);

  // ( - a ) * ( - b ) = a * b
  PROD_RULE(TYPT2, MINUS_S(A), MINUS_S(B), PROD(A, B), _1._ * _2._);

  // ( -a ) * b = -( a * b )
  PROD_RULE(TYPT2, MINUS_S(A), B, MINUS(PROD(A, B)), -(_1._ * _2));

  // a * ( b / c ) = ( a * b ) / c
  PROD_RULE(TYPT3, A, RATIO_S(B, C), RATIO(PROD(A, B), C), (_1 * _2._1)/_2._2);

  // n * ( a / b ) = ( n * a ) / b
  PROD_RULE(TYPN1T2, NUM(n), RATIO_S(A, B),
	    RATIO(PROD(NUM(n), A), B), (_1 * _2._1)/_2._2);

  // 1 * ( a / b ) = a / b
  PROD_RULE(TYPT2, NUM(1), RATIO_S(A, B), RATIO(A, B), _2);
    
  // 0 * ( a / b ) = 0
  PROD_RULE(TYPT2, NUM(0), RATIO_S(A, B), NUM(0), num<0>());

  // a * n = n * a
  PROD_RULE(TYPN1T1, A, NUM(n), PROD(NUM(n), A), _2 * _1);

  // ( a * b ) n = ( n * a ) * b
  PROD_RULE(TYPN1T2, PROD_S(A, B), NUM(n), PROD(PROD(NUM(n), A), B),
	    (_2 * _1._1) * _1._2);

  // ( a * b ) * ( c * d ) => ( ( a * b ) * c ) * d 
  PROD_RULE(TYPT4, PROD_S(A, B), PROD_S(C, D),
	     PROD(PROD(PROD(A, B), C), D),
	     (_1 * _2._1) * _2._2);
  
  // n/m * ( a / k ) = n/(m+k) * a
  PROD_RULE(TYPN3T1, FRACT_S(n, m), RATIO_S(A, NUM(k)),
	    PROD(FRACT(n, m + k), A), (fract<n, m + k>() * _2._1));

  // ( a / b ) * n = ( n a ) / b
  PROD_RULE(TYPN1T2, RATIO_S(A, B), NUM(n),
	     RATIO(PROD(NUM(n), A), B), (_2 * _1._1) / _1._2);

  // ( a / b ) * c = ( a * c ) / b
  PROD_RULE(TYPT3, RATIO_S(A, B), C,
	     RATIO(PROD(A, C), B), (_1._1 * _2) / _1._2);

  // 0 * 1 = 0  ( avoid template ambiguity )
  PROD_RULE(TYP0, NUM(0), NUM(1), NUM(0), num<0>());
    
  // ( a / b ) * ( c / d )= a * c / ( b * d )
  PROD_RULE(TYPT4, RATIO_S(A, B), RATIO_S(C, D),
	     RATIO(PROD(A, C), PROD(B, D)),
	     (_1._1 * _2._1)/(_1._2 * _2._2));

  // a^b * a^c => a^( b + c )
  template< TYPT3, bool parametric = Parametric<A>::value>
  struct SimplifyPowerProduct {
    typedef POWER( A, B ) arg1;
    typedef POWER( A, C ) arg2;
    typedef PROD_S( arg1, arg2 ) type;
    COMBINE( arg1, arg2, type( _1, _2 ) );
  };
  
  TEMPL( T3 )
  struct SimplifyPowerProduct< A, B, C, false > {
    typedef POWER( A, B ) arg1;
    typedef POWER( A, C ) arg2;
    typedef POWER( A, SUM( B, C ) ) type;
	 inline static type combine( const arg1 & _1, const arg2 & _2 ) 
    { return pow( DecomposePower< A, B >::getBase( _1 ), 
		  ( DecomposePower< A, B >::getExp( _1 ) + 
		    DecomposePower< A, C >::getExp( _2 ) ) ); }
  };
  
  TEMPL( T3 ) struct Product< POWER_S( A, B ),POWER_S( A, C ) > :
    public SimplifyPowerProduct< A, B, C > { };
  
  TEMPL( T2 ) struct Product< POWER_S( A, B ),POWER_S( A, B ) > :
    public SimplifyPowerProduct< A, B, B > { };
  
  TEMPL( T2 ) struct Product< A, POWER_S( A, B ) > : 
    public SimplifyPowerProduct< A, NUM( 1 ), B > { };

  TEMPL( N1T1 ) struct Product< A, POWER_S( A, NUM( n ) ) > : 
    public SimplifyPowerProduct< A, NUM( 1 ), NUM( n ) > { };
  
  TEMPL( T2 ) struct Product< POWER_S( A, B ), A > : 
    public SimplifyPowerProduct< A, B, NUM( 1 ) > { };

  TEMPL( N1T1 ) struct Product< POWER_S( A, NUM( n ) ), A > : 
    public SimplifyPowerProduct< A, NUM( n ), NUM( 1 ) > { };

  TEMPL( T1 ) struct Product< A, A > : 
    public SimplifyPowerProduct< A, NUM( 1 ), NUM( 1 ) > { };

  TEMPL( T2 ) struct Product< PROD_S( A, B ), PROD_S( A, B ) > : 
    public SimplifyPowerProduct< PROD( A, B ), NUM( 1 ), NUM( 1 ) > { };

  TEMPL( T1 ) struct Product< MINUS_S( A ), MINUS_S( A ) > : 
    public SimplifyPowerProduct< MINUS_S( A ), NUM( 1 ), NUM( 1 ) > { };


  // n * n = n ^ 2
  PROD_RULE(TYPN1, NUM(n), NUM(n), NUM(n*n), num<n*n>());

  // a/ b * ( c * d ) = ( a * c * d ) / b
  PROD_RULE(TYPT4, RATIO_S(A, B), PROD_S(C, D),
	     RATIO(PROD(PROD(A, C), D), B),
	     ((_1._1 * _2._1) * _2._2) / _1._2);

  // simplify f * g * h regardless of the order 
  template <typename Prod, bool simplify = Prod::value> struct AuxProduct {
    typedef PROD(typename Prod::AB, typename Prod::C) type;
    COMBINE(typename Prod::AB, typename Prod::C, _1 * _2);
  };
  
  template<typename Prod>  struct AuxProduct<Prod, false> {
    typedef PROD_S(typename Prod::AB, typename Prod::C) type;
    COMBINE(typename Prod::AB, typename Prod::C, type(_1, _2));
  };
  
  template<typename F, typename G, typename H>
  struct Product<PROD_S(F, G), H> {
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
      typedef PROD_S(A, B) base;
      typedef PROD(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return f; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return g; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    struct prod2 { 
      typedef G A; typedef H B; typedef F C;
      typedef PROD_S(A, B) base;
      typedef PROD(A, B) AB;
      inline static const A& a(const F& f, const G& g, const H& h) { return g; }
      inline static const B& b(const F& f, const G& g, const H& h) { return h; }
      inline static const C& c(const F& f, const G& g, const H& h) { return f; }
      enum { value = not ::std::is_same<AB, base>::value };
    };
    
    typedef typename 
      ::boost::mpl::if_ <prod1, prod1,
        typename ::boost::mpl::if_ <prod2, prod2, prod0>::type
      >::type prod;
    typedef typename AuxProduct<prod>::type type;
    inline static type combine(const ProductStruct<F, G>& fg, const H& h) {
      const F& f = fg._1;
      const G& g = fg._2;
      const typename prod::A & a = prod::a(f, g, h);
      const typename prod::B & b = prod::b(f, g, h);
      const typename prod::C & c = prod::c(f, g, h);
      return AuxProduct<prod>::combine(a * b, c); 
    }
  };

}

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
