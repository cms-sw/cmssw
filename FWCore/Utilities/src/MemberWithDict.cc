
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  MemberWithDict::MemberWithDict() :
    dataMember_() {
  }

  MemberWithDict::MemberWithDict(TDataMember* dataMember) :
    dataMember_(dataMember) {
  }

  std::string
  MemberWithDict::name() const {
    return dataMember_->GetName();
  }

  ObjectWithDict
  MemberWithDict::get() const {
    return (ObjectWithDict(typeOf(), reinterpret_cast<void*>(dataMember_->GetOffset())));
  }

  ObjectWithDict
  MemberWithDict::get(ObjectWithDict const& obj) const {
    return (ObjectWithDict(typeOf(), static_cast<char*>(obj.address()) + dataMember_->GetOffset()));
  }

  TypeWithDict
  MemberWithDict::typeOf() const {
    return TypeWithDict::byName(dataMember_->GetTypeName(), dataMember_->Property());
  }

  TypeWithDict
  MemberWithDict::declaringType() const {
    return TypeWithDict(dataMember_->GetClass(), dataMember_->Property());
  }

  bool
  MemberWithDict::isConst() const {
    return (dataMember_->Property() & kIsConstant);
  }

  bool
  MemberWithDict::isPublic() const {
    return (dataMember_->Property() & kIsPublic);
  }

  bool
  MemberWithDict::isStatic() const {
    return (dataMember_->Property() & kIsStatic);
  }

  bool
  MemberWithDict::isTransient() const {
    return !dataMember_->IsPersistent();
  }

  size_t
  MemberWithDict::offset() const {
    return (dataMember_->GetOffset());
  }

  MemberWithDict::operator bool() const {
    return (dataMember_ != nullptr);
  }

}
