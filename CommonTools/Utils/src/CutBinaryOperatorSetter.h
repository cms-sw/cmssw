#ifndef CommonTools_Utils_CutBinaryOperatorSetter_h
#define CommonTools_Utils_CutBinaryOperatorSetter_h
/* \class reco::parser::CutBinaryOperator
 *
 * Binary operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/CutBinaryOperator.h"
#include "CommonTools/Utils/src/CutStack.h"
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
