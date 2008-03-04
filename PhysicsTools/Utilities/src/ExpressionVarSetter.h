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

namespace reco {
  class MethodMap;

  namespace parser {
    struct ExpressionVarSetter {
      ExpressionVarSetter( ExpressionStack & stack, const reco::MethodMap & methods ) : 
	stack_( stack ), methods_( methods ) { }
      void operator()( const char *, const char * ) const;
    private:
      ExpressionStack & stack_;
      const MethodMap & methods_;
    };
  }
}

#endif
