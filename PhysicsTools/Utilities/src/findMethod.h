#ifndef PhysicsTools_Utilities_findMethod_h
#define PhysicsTools_Utilities_findMethod_h
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include <string>

namespace reco {
  // second pair member is true if a reference is found 
  // of type edm::Ref, edm::RefToBase or edm::Ptr
  std::pair<ROOT::Reflex::Member, bool> findMethod(const ROOT::Reflex::Type & type,
						   const std::string & name,
						   size_t args);
}

#endif
