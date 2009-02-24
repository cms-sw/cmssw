#ifndef CommonTools_Utils_ExpressionVar_h
#define CommonTools_Utils_ExpressionVar_h
/* \class reco::parser::ExpressionVar
 *
 * Variable expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/TypeCode.h"
#include "CommonTools/Utils/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar(const std::vector<MethodInvoker> & methods, method::TypeCode retType);
      virtual double value(const Reflex::Object & o) const;

      static bool isValidReturnType(method::TypeCode);
    private:
      std::vector<MethodInvoker> methods_;
      static void trueDelete(Reflex::Object & o) ;
      method::TypeCode retType_;
    }; 
  }
}

#endif
