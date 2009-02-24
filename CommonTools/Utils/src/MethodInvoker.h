#ifndef CommonTools_Utils_MethodInvoker_h
#define CommonTools_Utils_MethodInvoker_h
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "CommonTools/Utils/src/AnyMethodArgument.h"
#include <vector>

namespace reco {
  namespace parser {

    struct MethodInvoker {
      explicit MethodInvoker(const Reflex::Member & method,
			     const std::vector<AnyMethodArgument>    & ints   = std::vector<AnyMethodArgument>() );
      MethodInvoker(const MethodInvoker &); 
      /// Returns the object, and an info about if we have to delete such object or not
      std::pair<Reflex::Object,bool> value(const Reflex::Object & o) const;
      const Reflex::Member & method() const { return method_; }
      MethodInvoker & operator=(const MethodInvoker &);
    private:
      Reflex::Member method_;
      std::vector<AnyMethodArgument> ints_; // already fixed to the correct type
      std::vector<void*> args_;
      bool isFunction_;
      void setArgs();
    };
  }
}

#endif
