
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

  std::string
  MemberWithDict::name() const {
    return member_.Name();
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

  bool
  MemberWithDict::isConst() const {
    return member_.IsConst();
  }

  bool
  MemberWithDict::isPublic() const {
    return member_.IsPublic();
  }

  bool
  MemberWithDict::isStatic() const {
    return member_.IsStatic();
  }

  bool
  MemberWithDict::isTransient() const {
    return member_.IsTransient();
  }

  size_t
  MemberWithDict::offset() const {
    return member_.Offset();
  }

  MemberWithDict::operator bool() const {
    return bool(member_);
  }

}
