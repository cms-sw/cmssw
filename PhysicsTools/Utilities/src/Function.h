#ifndef Parser_Function_h
#define Parser_Function_h
/* \class reco::parser::Function
 *
 * Function enumerator
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#ifdef BOOST_SPIRIT_DEBUG 
#include <string>
#endif

namespace reco {
  namespace parser {    
    enum Function { 
      kAbs, kAcos, kAsin, kAtan, kAtan2, kChi2Prob, kCos, kCosh, kExp, 
      kLog, kLog10, kMax, kMin, kPow, kSin, kSinh, kSqrt, kTan, kTanh 
    };

#ifdef BOOST_SPIRIT_DEBUG 
  static const std::string functionNames[] = 
    { "abs", "acos", "asin", "atan", "atan2", "chi2prob", "cos", "cosh", "exp", 
      "log", "log10", "max", "min", "pow", "sin", "sinh", "sqrt", "tan", "tanh" };

#endif
  }
}

#endif
