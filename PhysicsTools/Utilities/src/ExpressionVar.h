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
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "PhysicsTools/Utilities/src/ExpressionBase.h"

namespace reco {
  namespace parser {
    struct ExpressionVar : public ExpressionBase {
      ExpressionVar( MethodMap::method_t m ): method_( m ) { }
      virtual double value( const ROOT::Reflex::Object & o ) const;
    private:
      MethodMap::method_t method_;
    }; 
  }
}

#endif
