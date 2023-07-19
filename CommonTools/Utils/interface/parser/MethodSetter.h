#ifndef CommonTools_Utils_MethodSetter_h
#define CommonTools_Utils_MethodSetter_h

/* \class reco::parser::MethodSetter
 *
 * \author Luca Lista, INFN
 *
 */

#include "CommonTools/Utils/interface/parser/MethodStack.h"
#include "CommonTools/Utils/interface/parser/TypeStack.h"
#include "CommonTools/Utils/interface/parser/MethodArgumentStack.h"

namespace reco {
  namespace parser {

    class MethodSetter {
    private:
      MethodStack& methStack_;
      LazyMethodStack& lazyMethStack_;
      TypeStack& typeStack_;
      MethodArgumentStack& intStack_;
      bool lazy_;

    public:
      explicit MethodSetter(MethodStack& methStack,
                            LazyMethodStack& lazyMethStack,
                            TypeStack& typeStack,
                            MethodArgumentStack& intStack,
                            bool lazy = false)
          : methStack_(methStack),
            lazyMethStack_(lazyMethStack),
            typeStack_(typeStack),
            intStack_(intStack),
            lazy_(lazy) {}

      void operator()(const char*, const char*) const;

      /// Resolve the name to either a function member or a data member,
      /// push a MethodInvoker on the MethodStack, and its return type to
      /// the TypeStack (after stripping "*" and "&").
      ///
      /// This method is used also by the LazyInvoker to perform the fetch once
      /// the final type is known.
      ///
      /// If the object is an edm::Ref/Ptr/RefToBase and the method is not
      /// found in the class:
      ///
      ///  1)  it pushes a no-argument 'get()' method
      ///
      ///  2)  if deep = true, it attempts to resolve and push the method on
      ///      the object to which the edm ref points to.  In that case, the
      ///      MethodStack will contain two more items after this call instead
      ///      of just one.  This behaviour is what you want for non-lazy parsing.
      ///
      ///  2b) if instead deep = false, it just pushes the 'get' on the stack.
      ///      this will allow the LazyInvoker to then re-discover the runtime
      ///      type of the pointee
      ///
      ///  The method will:
      ///     - throw exception, if the member can't be resolved
      ///     - return 'false' if deep = false and it only pushed
      //        a 'get' on the stack
      ///     - return true otherwise
      ///
      bool push(const std::string&, const std::vector<AnyMethodArgument>&, const char*, bool deep = true) const;
    };

  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_MethodSetter_h
