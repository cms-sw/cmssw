#ifndef Utilities_ExpressionVar_h
#define Utilities_ExpressionVar_h
/* \class reco::parser::ExpressionVar
 *
 * Variable expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
 #include "PhysicsTools/Utilities/src/ExpressionBase.h"
#include "PhysicsTools/Utilities/src/TypeCode.h"
#include "Reflex/Member.h"

namespace reco {
  namespace parser {
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar(const ROOT::Reflex::Member & method, method::TypeCode retType) : 
	method_(method), retType_(retType) { }
      virtual double value(const ROOT::Reflex::Object & o) const;

    private:
      ROOT::Reflex::Member method_;
      method::TypeCode retType_;
    }; 
  }
}

#endif
