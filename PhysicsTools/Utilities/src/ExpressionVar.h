#ifndef Utilities_ExpressionVar_h
#define Utilities_ExpressionVar_h
/* \class reco::parser::ExpressionVar
 *
 * Variable expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionBase.h"
#include "PhysicsTools/Utilities/src/TypeCode.h"
#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar(const std::vector<MethodInvoker> & methods, method::TypeCode retType);
      virtual double value(const ROOT::Reflex::Object & o) const;

    private:
      std::vector<MethodInvoker> methods_;
      method::TypeCode retType_;
    }; 
  }
}

#endif
