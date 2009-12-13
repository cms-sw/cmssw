#ifndef CommonTools_Utils_MethodSetter_h
#define CommonTools_Utils_MethodSetter_h
/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MethodSetter.h,v 1.5 2008/08/15 17:54:57 chrjones Exp $
 */
#include "CommonTools/Utils/src/MethodStack.h"
#include "CommonTools/Utils/src/TypeStack.h"
#include "CommonTools/Utils/src/MethodArgumentStack.h"

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
      void push(const std::string&, const std::vector<AnyMethodArgument>&,const char*) const;
    };
  }
}

#endif
