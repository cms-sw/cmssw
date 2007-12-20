#ifndef Utilities_ExpressionVarSetter_h
#define Utilities_ExpressionVarSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#include "PhysicsTools/Utilities/src/MethodStack.h"
#include "PhysicsTools/Utilities/src/TypeStack.h"

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
