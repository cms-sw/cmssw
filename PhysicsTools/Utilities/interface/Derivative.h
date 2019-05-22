#ifndef PhysicsTools_Utilities_Derivative_h
#define PhysicsTools_Utilities_Derivative_h

#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include <type_traits>

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  template<typename X, typename A>
    struct Derivative {
    typedef NUM(0) type;
    GET(A, num<0>());
  };

  TEMPL(XT1) DERIV(X, A) derivative(const A& _) { 
    return Derivative<X, A>::get(_); 
  }

  TEMPL(XT1) struct Independent : 
    public ::std::is_same<DERIV(X, A), NUM(0)> { };
  
  // dx / dx = 1 
  DERIV_RULE(TYPX, X, NUM(1), num<1>());

  // d exp(x) / dx = exp(x)
  DERIV_RULE(TYPXT1, EXP_S(A), PROD(EXP(A), DERIV(X, A)), _ * derivative<X>(_._));
  
  // d log(x) / dx = 1 / x
  DERIV_RULE(TYPXT1, LOG_S(A), PROD(RATIO(NUM(1), A), DERIV(X, A)),
	     (num<1>() / _._) * derivative<X>(_._));
  
  // d abs(x) / dx = sgn(x)
  DERIV_RULE(TYPXT1, ABS_S(A), PROD(SGN(A), DERIV(X, A)),
	     sgn(_._) * derivative<X>(_._));

  // d sin(x) / dx = cos(x)
  DERIV_RULE(TYPXT1, SIN_S(A), PROD(COS(A), DERIV(X, A)),
	     cos(_._) * derivative<X>(_._));

  // d cos(x) / dx = - sin(x)
  DERIV_RULE(TYPXT1, COS_S(A), MINUS(PROD(SIN(A), DERIV(X, A))),
	     - (sin(_._) * derivative<X>(_._)));

  // d tan(x) / dx = 1 / cos(x)^2
  DERIV_RULE(TYPXT1, TAN_S(A), PROD(RATIO(NUM(1), SQUARE(COS(A))), 
				    DERIV(X, A)),
	     (num<1>() / sqr(cos(_._))) * derivative<X>(_._));
  
  // d/dx (f + g) = d/dx f + d/dx g
  DERIV_RULE(TYPXT2, SUM_S(A, B), SUM(DERIV(X, A), DERIV(X, B)),
	     derivative<X>(_._1) + derivative<X>(_._2));
  
  // d/dx (-f) = - d/dx f
  DERIV_RULE(TYPXT1, MINUS_S(A), MINUS(DERIV(X, A)),
	     - derivative<X>(_._));

  // d/dx (f * g) =  d/dx f * g + f * d/dx g
  DERIV_RULE(TYPXT2, PROD_S(A, B), SUM(PROD(DERIV(X, A), B),
				       PROD(A, DERIV(X, B))),
	     derivative<X>(_._1) * _._2 + _._1 * derivative<X>(_._2));
  
  // d/dx (f / g) =  (d/dx f * g - f * d/dx g) / g^2
  DERIV_RULE(TYPXT2, RATIO_S(A, B),
	     RATIO(DIFF(PROD(DERIV(X, A), B),
			PROD(A, DERIV(X, B))),
		   SQUARE(B)),
	     (derivative<X>(_._1) * _._2 -
	      _._1 * derivative<X>(_._2)) / sqr(_._2));
  
  // d/dx f ^ n  = n f ^ (n - 1) d/dx f
  DERIV_RULE(TYPXN1T1, POWER_S(A, NUM(n)),
	     PROD(PROD(NUM(n), POWER(A, NUM(n - 1))), DERIV(X, A)),
	     _._2 * pow(_._1, num<n - 1>()) * derivative<X>(_._1));
    
  // d/dx f ^ n/m  = n/m f ^ (n/m - 1) d/dx f
  DERIV_RULE(TYPXN2T1, POWER_S(A, FRACT_S(n, m)),
	     PROD(PROD(FRACT(n, m), POWER(A, FRACT(n - m, n))),
		  DERIV(X, A)),
	     _._2 * pow(_._1, fract<n - m, m>()) * derivative<X>(_._1));
  
  // d sqrt(x) / dx =  1/2 1/sqrt(x)
  DERIV_RULE(TYPXT1, SQRT_S(A),
	     PROD(PROD(FRACT(1, 2), RATIO(NUM(1), SQRT(A))),
		  DERIV(X, A)),
	     (fract<1, 2>() * (num<1>() / sqrt(_._))) *
	     derivative<X>(_._));
}

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
