#ifndef PhysicsTools_Utilities_MethodInvoker_h
#define PhysicsTools_Utilities_MethodInvoker_h
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "PhysicsTools/Utilities/src/AnyMethodArgument.h"
#include <vector>

namespace reco {
  namespace parser {

    struct MethodInvoker {
      explicit MethodInvoker(const Reflex::Member & method,
			     const std::vector<AnyMethodArgument>    & ints   = std::vector<AnyMethodArgument>() );
      MethodInvoker(const MethodInvoker &); 

      /// Invokes the method, putting the result in retval.
      /// Returns the Object that points to the result value, after removing any "*" and "&" 
      /// Caller code is responsible for allocating retstore before calling 'invoke', and of deallocating it afterwards
      Reflex::Object
      invoke(const Reflex::Object & o, Reflex::Object &retstore) const;
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
