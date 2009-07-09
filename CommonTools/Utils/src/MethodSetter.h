#ifndef CommonTools_Utils_MethodSetter_h
#define CommonTools_Utils_MethodSetter_h
/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MethodSetter.h,v 1.1 2009/02/24 14:10:21 llista Exp $
 */
#include "CommonTools/Utils/src/MethodStack.h"
#include "CommonTools/Utils/src/TypeStack.h"
#include "CommonTools/Utils/src/MethodArgumentStack.h"

namespace reco {
  namespace parser {
    struct MethodSetter {
      explicit MethodSetter(MethodStack & methStack, LazyMethodStack & lazyMethStack, TypeStack & typeStack,
			    MethodArgumentStack & intStack, bool lazy=false) : 
	methStack_(methStack), lazyMethStack_(lazyMethStack), typeStack_(typeStack),
	intStack_(intStack), lazy_(lazy) { }
      void operator()(const char *, const char *) const;
      /// Resolve the method, push a MethodInvoker on the MethodStack and it's return type to TypeStack (after stripping "*" and "&")
      /// If the object is a Ref/Ptr/RefToBase and the method is not found in that class, it pushes a no-argument 'get()' method
      /// and attempts to resolve and push the method on the object to which the edm ref points to. In that case, the MethodStack will 
      /// contain two more items after this call instead of just one.
      /// This method is used also by the LazyInvoker to perform the fetch once the final type is known
      void push(const std::string&, const std::vector<AnyMethodArgument>&,const char*) const;
    private:
      MethodStack & methStack_;
      LazyMethodStack & lazyMethStack_;
      TypeStack & typeStack_;
      MethodArgumentStack & intStack_;
      bool lazy_;
    };
  }
}

#endif
