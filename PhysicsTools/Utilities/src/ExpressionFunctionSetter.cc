#include "PhysicsTools/Utilities/src/ExpressionFunctionSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionUnaryOperator.h"
#include "PhysicsTools/Utilities/src/ExpressionBinaryOperator.h"
#include <cmath>
#ifdef BOOST_SPIRIT_DEBUG 
#include <string>
#include <iostream>
#endif

namespace reco {
  namespace parser {
    struct abs_f { double operator()( double x ) const { return fabs( x ); } };
    struct acos_f { double operator()( double x ) const { return acos( x ); } };
    struct asin_f { double operator()( double x ) const { return asin( x ); } };
    struct atan_f { double operator()( double x ) const { return atan( x ); } };
    struct atan2_f { double operator()( double x, double y ) const { return atan2( x, y ); } };
    struct cos_f { double operator()( double x ) const { return cos( x ); } };
    struct cosh_f { double operator()( double x ) const { return cosh( x ); } };
    struct exp_f { double operator()( double x ) const { return exp( x ); } };
    struct log_f { double operator()( double x ) const { return log( x ); } };
    struct log10_f { double operator()( double x ) const { return log10( x ); } };
    struct pow_f { double operator()( double x, double y ) const { return pow( x, y ); } };
    struct sin_f { double operator()( double x ) const { return sin( x ); } };
    struct sinh_f { double operator()( double x ) const { return sinh( x ); } };
    struct sqrt_f { double operator()( double x ) const { return sqrt( x ); } };
    struct tan_f { double operator()( double x ) const { return tan( x ); } };
    struct tanh_f { double operator()( double x ) const { return tanh( x ); } };
  }
}

using namespace reco::parser;

void ExpressionFunctionSetter::operator()( const char *, const char * ) const {
  Function fun = funStack_.back(); funStack_.pop_back();
#ifdef BOOST_SPIRIT_DEBUG 
  BOOST_SPIRIT_DEBUG_OUT << "pushing function expression: " << functionNames[ fun ] << std::endl;
#endif
  ExpressionPtr funExp;
  switch( fun ) {
  case( kAbs   ) : funExp.reset( new ExpressionUnaryOperator <abs_f  >( expStack_ ) ); break;
  case( kAcos  ) : funExp.reset( new ExpressionUnaryOperator <acos_f >( expStack_ ) ); break;
  case( kAsin  ) : funExp.reset( new ExpressionUnaryOperator <asin_f >( expStack_ ) ); break;
  case( kAtan  ) : funExp.reset( new ExpressionUnaryOperator <atan_f >( expStack_ ) ); break;
  case( kAtan2 ) : funExp.reset( new ExpressionBinaryOperator<atan2_f>( expStack_ ) ); break;
  case( kCos   ) : funExp.reset( new ExpressionUnaryOperator <cos_f  >( expStack_ ) ); break;
  case( kCosh  ) : funExp.reset( new ExpressionUnaryOperator <cosh_f >( expStack_ ) ); break;
  case( kExp   ) : funExp.reset( new ExpressionUnaryOperator <exp_f  >( expStack_ ) ); break;
  case( kLog   ) : funExp.reset( new ExpressionUnaryOperator <log_f  >( expStack_ ) ); break;
  case( kLog10 ) : funExp.reset( new ExpressionUnaryOperator <log10_f>( expStack_ ) ); break;
  case( kPow   ) : funExp.reset( new ExpressionBinaryOperator<pow_f  >( expStack_ ) ); break;
  case( kSin   ) : funExp.reset( new ExpressionUnaryOperator <sin_f  >( expStack_ ) ); break;
  case( kSinh  ) : funExp.reset( new ExpressionUnaryOperator <sinh_f >( expStack_ ) ); break;
  case( kSqrt  ) : funExp.reset( new ExpressionUnaryOperator <sqrt_f >( expStack_ ) ); break;
  case( kTan   ) : funExp.reset( new ExpressionUnaryOperator <tan_f  >( expStack_ ) ); break;
  case( kTanh  ) : funExp.reset( new ExpressionUnaryOperator <tanh_f >( expStack_ ) ); break;
  };
  expStack_.push_back( funExp );
}
