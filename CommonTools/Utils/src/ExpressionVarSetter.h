#ifndef CommonTools_Utils_ExpressionVarSetter_h
#define CommonTools_Utils_ExpressionVarSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
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
			  LazyMethodStack & lazyMethStack, 
			  TypeStack & typeStack) : 
	exprStack_(exprStack), 
	methStack_(methStack),
	lazyMethStack_(lazyMethStack),
	typeStack_(typeStack) { }
      void operator()(const char *, const char *) const;

    private:
      void push(const char *, const char *) const;
      void lazyPush(const char *, const char *) const;

      ExpressionStack & exprStack_;
      MethodStack & methStack_;
      LazyMethodStack & lazyMethStack_;
      TypeStack & typeStack_;
    };
  }
}

#endif
