#include "CommonTools/Utils/src/ExpressionFunctionSetter.h"
#include "CommonTools/Utils/src/ExpressionUnaryOperator.h"
#include "CommonTools/Utils/src/ExpressionBinaryOperator.h"
#include "CommonTools/Utils/src/ExpressionQuaterOperator.h"
#include <cmath>
#include <Math/ProbFuncMathCore.h>
#include <DataFormats/Math/interface/deltaPhi.h>
#include <DataFormats/Math/interface/deltaR.h>

namespace reco {
  namespace parser {
    struct abs_f { double operator()( double x ) const { return fabs( x ); } };
    struct acos_f { double operator()( double x ) const { return acos( x ); } };
    struct asin_f { double operator()( double x ) const { return asin( x ); } };
    struct atan_f { double operator()( double x ) const { return atan( x ); } };
    struct atan2_f { double operator()( double x, double y ) const { return atan2( x, y ); } };
    struct chi2prob_f { double operator()( double x, double y ) const { return ROOT::Math::chisquared_cdf_c( x, y ); } };
    struct cos_f { double operator()( double x ) const { return cos( x ); } };
    struct cosh_f { double operator()( double x ) const { return cosh( x ); } };
    struct deltaR_f { double operator()( double e1, double p1, double e2, double p2 ) const { return reco::deltaR(e1,p1,e2,p2); } };
    struct deltaPhi_f { double operator()( double p1, double p2 ) const { return reco::deltaPhi(p1,p2); } };
    struct exp_f { double operator()( double x ) const { return exp( x ); } };
    struct hypot_f { double operator()( double x, double y ) const { return hypot( x, y ); } };
    struct log_f { double operator()( double x ) const { return log( x ); } };
    struct log10_f { double operator()( double x ) const { return log10( x ); } };
    struct max_f { double operator()( double x, double y ) const { return std::max( x, y ); } };
    struct min_f { double operator()( double x, double y ) const { return std::min( x, y ); } };
    struct pow_f { double operator()( double x, double y ) const { return pow( x, y ); } };
    struct sin_f { double operator()( double x ) const { return sin( x ); } };
    struct sinh_f { double operator()( double x ) const { return sinh( x ); } };
    struct sqrt_f { double operator()( double x ) const { return sqrt( x ); } };
    struct tan_f { double operator()( double x ) const { return tan( x ); } };
    struct tanh_f { double operator()( double x ) const { return tanh( x ); } };
    struct test_bit_f { double operator()( double mask, double iBit ) const { return (int(mask) >> int(iBit)) & 1; } };
  }
}

using namespace reco::parser;

void ExpressionFunctionSetter::operator()( const char *, const char * ) const {
  Function fun = funStack_.back(); funStack_.pop_back();
  ExpressionPtr funExp;
  switch( fun ) {
  case( kAbs      ) : funExp.reset( new ExpressionUnaryOperator <abs_f  >   ( expStack_ ) ); break;
  case( kAcos     ) : funExp.reset( new ExpressionUnaryOperator <acos_f >   ( expStack_ ) ); break;
  case( kAsin     ) : funExp.reset( new ExpressionUnaryOperator <asin_f >   ( expStack_ ) ); break;
  case( kAtan     ) : funExp.reset( new ExpressionUnaryOperator <atan_f >   ( expStack_ ) ); break;
  case( kAtan2    ) : funExp.reset( new ExpressionBinaryOperator<atan2_f>   ( expStack_ ) ); break;
  case( kChi2Prob ) : funExp.reset( new ExpressionBinaryOperator<chi2prob_f>( expStack_ ) ); break;
  case( kCos      ) : funExp.reset( new ExpressionUnaryOperator <cos_f  >   ( expStack_ ) ); break;
  case( kCosh     ) : funExp.reset( new ExpressionUnaryOperator <cosh_f >   ( expStack_ ) ); break;
  case( kDeltaR   ) : funExp.reset( new ExpressionQuaterOperator<deltaR_f>  ( expStack_ ) ); break;
  case( kDeltaPhi ) : funExp.reset( new ExpressionBinaryOperator<deltaPhi_f>( expStack_ ) ); break;
  case( kExp      ) : funExp.reset( new ExpressionUnaryOperator <exp_f  >   ( expStack_ ) ); break;
  case( kHypot    ) : funExp.reset( new ExpressionBinaryOperator<hypot_f>   ( expStack_ ) ); break;
  case( kLog      ) : funExp.reset( new ExpressionUnaryOperator <log_f  >   ( expStack_ ) ); break;
  case( kLog10    ) : funExp.reset( new ExpressionUnaryOperator <log10_f>   ( expStack_ ) ); break;
  case( kMax      ) : funExp.reset( new ExpressionBinaryOperator<max_f>     ( expStack_ ) ); break;
  case( kMin      ) : funExp.reset( new ExpressionBinaryOperator<min_f>     ( expStack_ ) ); break;
  case( kPow      ) : funExp.reset( new ExpressionBinaryOperator<pow_f  >   ( expStack_ ) ); break;
  case( kSin      ) : funExp.reset( new ExpressionUnaryOperator <sin_f  >   ( expStack_ ) ); break;
  case( kSinh     ) : funExp.reset( new ExpressionUnaryOperator <sinh_f >   ( expStack_ ) ); break;
  case( kSqrt     ) : funExp.reset( new ExpressionUnaryOperator <sqrt_f >   ( expStack_ ) ); break;
  case( kTan      ) : funExp.reset( new ExpressionUnaryOperator <tan_f  >   ( expStack_ ) ); break;
  case( kTanh     ) : funExp.reset( new ExpressionUnaryOperator <tanh_f >   ( expStack_ ) ); break;
  case( kTestBit  ) : funExp.reset( new ExpressionBinaryOperator<test_bit_f>( expStack_ ) ); break;
  };
  expStack_.push_back( funExp );
}
