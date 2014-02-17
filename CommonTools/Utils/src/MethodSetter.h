#ifndef CommonTools_Utils_MethodSetter_h
#define CommonTools_Utils_MethodSetter_h
/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MethodSetter.h,v 1.4 2012/06/26 21:13:13 wmtan Exp $
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
      /// This method is used also by the LazyInvoker to perform the fetch once the final type is known
      /// If the object is a Ref/Ptr/RefToBase and the method is not found in that class:
      ///  1)  it pushes a no-argument 'get()' method
      ///  2)  if deep = true, it attempts to resolve and push the method on the object to which the edm ref points to. 
      ///         In that case, the MethodStack will contain two more items after this call instead of just one.
      ///         This behaviour is what you want for non-lazy parsing
      ///  2b) if instead deep = false, it just pushes the 'get' on the stack.
      ///      this will allow the LazyInvoker to then re-discover the runtime type of the pointee
      ///  The method will:
      ///     - throw exception, if the member can't be resolved
      ///     - return 'false' if deep = false and it only pushed a 'get' on the stack
      ///     - return true otherwise
      bool push(const std::string&, const std::vector<AnyMethodArgument>&,const char*,bool deep=true) const;
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
