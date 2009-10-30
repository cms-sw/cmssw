#ifndef CommonTools_Utils_ExpressionVarSetter_h
#define CommonTools_Utils_ExpressionVarSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/ExpressionStack.h"
#include "CommonTools/Utils/src/MethodStack.h"
#include "CommonTools/Utils/src/TypeStack.h"

namespace reco {
  namespace parser {
    struct ExpressionVarSetter {
      ExpressionVarSetter(ExpressionStack & exprStack, 
			  MethodStack & methStack, 
			  TypeStack & typeStack) : 
	exprStack_(exprStack), 
	methStack_(methStack),
	typeStack_(typeStack) { }
      void operator()(const char *, const char *) const;
    private:
      ExpressionStack & exprStack_;
      MethodStack & methStack_;
      TypeStack & typeStack_;
    };
  }
}

#endif
