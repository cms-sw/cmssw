#ifndef PhysicsTools_Utilities_SimplifyMinus_h
#define PhysicsTools_Utilities_SimplifyMinus_h

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
namespace funct {

  // - - a = a
  MINUS_RULE(TYPT1, MINUS_S(A), A, _._);

  // -( a + b ) = ( - a ) + ( -b )
  MINUS_RULE(TYPT2, SUM_S(A, B), SUM(MINUS(A), MINUS(B)), (-_._1) + (-_._2));

}  // namespace funct

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
