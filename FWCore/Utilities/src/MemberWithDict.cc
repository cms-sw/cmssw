
#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {
  ObjectWithDict
  MemberWithDict::get() const {
    return (ObjectWithDict(member_.Get()));
  }

  ObjectWithDict
  MemberWithDict::get(ObjectWithDict const& obj) const {
    return (ObjectWithDict(member_.Get(obj.object_)));
  }

  TypeWithDict
  MemberWithDict::typeOf() const {
    return (TypeWithDict(member_.TypeOf()));
  }

  TypeWithDict
  MemberWithDict::declaringType() const {
    return (TypeWithDict(member_.DeclaringType()));
  }
}
