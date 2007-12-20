#ifndef PhysicsTools_Utilities_MethodSetter_h
#define PhysicsTools_Utilities_MethodSetter_h
/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 */
#include "PhysicsTools/Utilities/src/MethodStack.h"
#include "PhysicsTools/Utilities/src/TypeStack.h"

namespace reco {
  namespace parser {
    struct MethodSetter {
      explicit MethodSetter(MethodStack & methStack, TypeStack & typeStack) : 
	methStack_(methStack), typeStack_(typeStack) { }
      void operator()(const char *, const char *) const;
    private:
      MethodStack & methStack_;
      TypeStack & typeStack_;
    };
  }
}

#endif
