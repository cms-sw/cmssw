#ifndef Utilities_ExpressionVarSetter_h
#define Utilities_ExpressionVarSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#include "Reflex/Type.h"

namespace reco {
  namespace parser {
    struct ExpressionVarSetter {
      ExpressionVarSetter(ExpressionStack & stack, const ROOT::Reflex::Type & type) : 
	stack_(stack), type_(type) { }
      void operator()(const char *, const char *) const;
    private:
      ExpressionStack & stack_;
      ROOT::Reflex::Type type_;
    };
  }
}

#endif
