#ifndef Utilities_CutBinaryOperatorSetter_h
#define Utilities_CutBinaryOperatorSetter_h
/* \class reco::parser::CutBinaryOperator
 *
 * Binary operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/CutBinaryOperator.h"
#include "PhysicsTools/Utilities/src/CutStack.h"
#include <cmath>

namespace reco {
  namespace parser {

    template<typename T>
    struct power_of {
      T operator()(T lhs, T rhs) const { return pow(lhs, rhs); }
    };

    template<typename Op>
    struct CutBinaryOperatorSetter {
      CutBinaryOperatorSetter(CutStack & stack) : stack_(stack) { }
      void operator()(const char*, const char*) const {
	stack_.push_back(CutPtr(new CutBinaryOperator<Op>(stack_)));
      }
    private:
      CutStack & stack_;
    };
  }
}

#endif
