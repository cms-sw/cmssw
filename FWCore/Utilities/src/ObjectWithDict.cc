#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/ObjectWithDict.h"

namespace edm {

  ObjectWithDict::ObjectWithDict() :
    object_(),
    type_(),
    address_(nullptr) {
  }

  ObjectWithDict::ObjectWithDict(Reflex::Object const& obj) :
    object_(obj),
    type_(obj.TypeOf()),
    address_(obj.Address()) {
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type) :
    object_(type.type_.Construct()),
    type_(type),
    address_(object_.Address()) {
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type,
                                 TypeWithDict const& signature,
                                 std::vector<void*> const& values) :
    object_(type.type_.Construct(signature.type_, values)),
    type_(type),
    address_(object_.Address()) {
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type, void* address) :
    object_(type.type_, address),
    type_(type),
    address_(address) {
  }

  ObjectWithDict::ObjectWithDict(std::type_info const& typeID, void* address) :
    object_(Reflex::Type::ByTypeInfo(typeID), address),
    type_(TypeWithDict(typeID)),
    address_(address) {
  }

  std::string
  ObjectWithDict::typeName() const {
    return type_.name();
  }

  bool
  ObjectWithDict::isPointer() const {
    return type_.isPointer();
  }

  bool
  ObjectWithDict::isReference() const {
    return type_.isReference();
  }

  bool
  ObjectWithDict::isTypedef() const {
    return type_.isTypedef();
  }

  TypeWithDict
  ObjectWithDict::typeOf() const {
    return type_;
  }

  TypeWithDict
  ObjectWithDict::toType() const {
    return TypeWithDict(type_.toType());
  }

  TypeWithDict
  ObjectWithDict::finalType() const {
    return TypeWithDict(type_.finalType());
  }

  TypeWithDict
  ObjectWithDict::dynamicType() const {
    return TypeWithDict(object_.DynamicType());
  }

  ObjectWithDict
  ObjectWithDict::castObject(TypeWithDict const& type) const {
    return ObjectWithDict(object_.CastObject(type.type_));
  }

  ObjectWithDict
  ObjectWithDict::construct() const {
    return ObjectWithDict(object_.TypeOf().Construct());
  }

  void
  ObjectWithDict::destruct() const {
    object_.Destruct();
  }

  ObjectWithDict::operator bool() const {
    return bool(type_) && address_ != nullptr;
  }

  void*
  ObjectWithDict::address() const {
    return address_;
  }

  void
  ObjectWithDict::invoke(std::string const& fm, ObjectWithDict* ret) const{
    object_.Invoke(fm, &ret->object_);
  }

  ObjectWithDict
  ObjectWithDict::get(std::string const& member) const {
    return ObjectWithDict(object_.Get(member));
  }
}
