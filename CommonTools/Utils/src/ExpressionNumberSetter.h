#ifndef CommonTools_Utils_ExpressionNumberSetter_h
#define CommonTools_Utils_ExpressionNumberSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/ExpressionNumber.h"
#include "CommonTools/Utils/src/ExpressionStack.h"

namespace reco {
  namespace parser {
    struct ExpressionNumberSetter {
      ExpressionNumberSetter(ExpressionStack & stack) : stack_(stack) { }
      void operator()(double n) const {
	stack_.push_back(ExpressionPtr(new ExpressionNumber(n)));
      }
    private:
      ExpressionStack & stack_;
    };
  }
}

#endif
