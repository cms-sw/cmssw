#ifndef PhysicsTools_Utilities_SimplifyPower_h
#define PhysicsTools_Utilities_SimplifyPower_h

#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Ratio.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Sqrt.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/DecomposePower.h"

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  // a ^ 1 = a
  POWER_RULE(TYPT1, A, NUM(1), A, _1);

  // a ^ -1 = 1 / a
  POWER_RULE(TYPT1, A, NUM(-1), RATIO(NUM(1), A), num<1>() / _1);

  // a ^ 1/2 =  sqrt(a)
  POWER_RULE(TYPT1, A, FRACT_S(1, 2), SQRT(A), sqrt(_1));

  // a ^ 0 = 1
  POWER_RULE(TYPT1, A, NUM(0), NUM(1), num<1>());

  // (a * b)^ 0 = 1
  POWER_RULE(TYPT2, PROD_S(A, B), NUM(0), NUM(1), num<1>());

  // (a ^ b) ^ c = a ^ (b + c)
  POWER_RULE(TYPT3, POWER_S(A, B), C, POWER(A, SUM(B, C)), pow(_1._1, _1._2 + _2));

  // (a ^ b) ^ n = a ^ (b + n)
  POWER_RULE(TYPN1T2, POWER_S(A, B), NUM(n), POWER(A, SUM(B, NUM(n))), pow(_1._1, _1._2 + _2));

  // a ^ (-n) = 1 / a ^ n
  template <TYPN1T1, bool positive = (n >= 0)>
  struct SimplifySignedPower {
    typedef POWER_S(A, NUM(n)) type;
    COMBINE(A, NUM(n), type(_1, _2));
  };

  TEMPL(N1T1) struct SimplifySignedPower<n, A, false> {
    typedef RATIO(NUM(1), POWER(A, NUM(-n))) type;
    COMBINE(A, NUM(n), num<1>() / pow(_1, num<-n>()));
  };

  TEMPL(T1) struct SimplifySignedPower<0, A, true> {
    typedef NUM(1) type;
    COMBINE(A, NUM(0), num<1>());
  };

  TEMPL(N1T1) struct Power<A, NUM(n)> : public SimplifySignedPower<n, A> {};

}  // namespace funct

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
