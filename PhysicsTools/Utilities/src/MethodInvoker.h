#ifndef PhysicsTools_Utilities_MethodInvoker_h
#define PhysicsTools_Utilities_MethodInvoker_h
#include "Reflex/Object.h"
#include "Reflex/Member.h"

namespace reco {
  namespace parser {
    struct MethodInvoker {
      explicit MethodInvoker(const ROOT::Reflex::Member & method) :
	method_(method) { }
      ROOT::Reflex::Object value(const ROOT::Reflex::Object & o) const;
      const ROOT::Reflex::Member & method() const { return method_; }
    private:
      ROOT::Reflex::Member method_;
    };
  }
}

#endif
