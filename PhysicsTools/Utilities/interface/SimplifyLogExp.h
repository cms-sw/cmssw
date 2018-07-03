#ifndef PhysicsTools_Utilities_SimplifyLogExp_h
#define PhysicsTools_Utilities_SimplifyLogExp_h

#include "PhysicsTools/Utilities/interface/Log.h"
#include "PhysicsTools/Utilities/interface/Exp.h"
#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Difference.h"
#include "PhysicsTools/Utilities/interface/Ratio.h"

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {
  // log(exp(a)) = a
  LOG_RULE(TYPT1, EXP_S(A), A, _._);
  
  // exp(log(a)) = a
  EXP_RULE(TYPT1, LOG_S(A), A, _._);
    
  // log(a ^ b) = b * log(a)
  LOG_RULE(TYPT2, POWER_S(A, B), PROD(B, LOG(A)), _._2 * log(_._1));
  
  // exp(a) * exp(b) = exp(a + b)
  PROD_RULE(TYPT2, EXP_S(A), EXP_S(B), EXP(SUM(A, B)), exp(_1._ + _2._));

  // log(a * b) = log(a) + log (b)
  LOG_RULE(TYPT2, PROD_S(A, B), SUM(LOG(A), LOG(B)), log(_._1) + log(_._2));
    
  // log(a / b) = log(a) - log (b)
  LOG_RULE(TYPT2, RATIO_S(A, B), DIFF(LOG(A), LOG(B)), log(_._1) - log(_._2));
  
  // exp(x) * x = x * exp(x)
  PROD_RULE(TYPT1, EXP_S(A), A, PROD(A, EXP(A)), _2 * _1 );

  // log(x) * x = x * log(x)
  PROD_RULE(TYPT1, LOG_S(A), A, PROD(A, LOG(A)), _2 * _1 );

}

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
