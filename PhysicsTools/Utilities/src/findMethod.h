#ifndef PhysicsTools_Utilities_findMethod_h
#define PhysicsTools_Utilities_findMethod_h
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include <string>

namespace reco {
  ROOT::Reflex::Member findMethod(const ROOT::Reflex::Type & type,
				  const std::string & name);
  void checkMethod(const ROOT::Reflex::Type & type, 
		   const ROOT::Reflex::Member & mem);
}

#endif
