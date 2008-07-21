#ifndef PhysicsTools_Utilities_MethodSetter_h
#define PhysicsTools_Utilities_MethodSetter_h
/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MethodSetter.h,v 1.3 2008/01/08 10:14:33 llista Exp $
 */
#include "PhysicsTools/Utilities/src/MethodStack.h"
#include "PhysicsTools/Utilities/src/TypeStack.h"
#include "PhysicsTools/Utilities/src/MethodArgumentStack.h"

namespace reco {
  namespace parser {
    struct MethodSetter {
      explicit MethodSetter(MethodStack & methStack, TypeStack & typeStack,
			    MethodArgumentStack & intStack) : 
	methStack_(methStack), typeStack_(typeStack),
	intStack_(intStack) { }
      void operator()(const char *, const char *) const;
    private:
      MethodStack & methStack_;
      TypeStack & typeStack_;
      MethodArgumentStack & intStack_;
      void push(const std::string&, const std::vector<AnyMethodArgument>&) const;
    };
  }
}

#endif
