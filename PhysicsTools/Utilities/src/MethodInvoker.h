#ifndef PhysicsTools_Utilities_MethodInvoker_h
#define PhysicsTools_Utilities_MethodInvoker_h
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "PhysicsTools/Utilities/src/AnyMethodArgument.h"
#include <vector>

namespace reco {
  namespace parser {

    struct MethodInvoker {
      explicit MethodInvoker(const ROOT::Reflex::Member & method,
			     const std::vector<AnyMethodArgument>    & ints   = std::vector<AnyMethodArgument>() );
      MethodInvoker(const MethodInvoker &); 
      ROOT::Reflex::Object value(const ROOT::Reflex::Object & o) const;
      const ROOT::Reflex::Member & method() const { return method_; }
      MethodInvoker & operator=(const MethodInvoker &);
    private:
      ROOT::Reflex::Member method_;
      std::vector<AnyMethodArgument> ints_; // already fixed to the correct type
      std::vector<void*> args_;
      void setArgs();
    };
  }
}

#endif
