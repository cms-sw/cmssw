#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

namespace edm {

  ObjectWithDict::ObjectWithDict() :
    type_(),
    address_(nullptr) {
  }

  ObjectWithDict
  ObjectWithDict::byType(TypeWithDict const& type) {
    ObjectWithDict obj(type.construct());
    return obj;
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type, void* address) :
    type_(type),
    address_(address) {
  }

  ObjectWithDict::ObjectWithDict(std::type_info const& typeID, void* address) :
    type_(TypeWithDict(typeID)),
    address_(address) {
  }

  TypeWithDict const&
  ObjectWithDict::typeOf() const {
    return type_;
  }

  TypeWithDict
  ObjectWithDict::dynamicType() const {
    if(!type_.isVirtual()) {
      return type_;
    }
    struct Dummy_t {virtual ~Dummy_t() {} };
    return TypeWithDict(typeid(*(Dummy_t*)address_));
  }

  ObjectWithDict::operator bool() const {
    return bool(type_) && address_ != nullptr;
  }

  void*
  ObjectWithDict::address() const {
    return address_;
  }

  ObjectWithDict
  ObjectWithDict::get(std::string const& memberName) const {
    return type_.dataMemberByName(memberName).get(*this);
  }
}
