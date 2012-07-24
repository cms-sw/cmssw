#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type) : object_(type.type_.Construct()) {}

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type,
                                 TypeWithDict const& signature,
                                 std::vector<void*> const& values) : object_(type.type_.Construct(signature.type_, values)) {}

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type, void* address) : object_(type.type_, address) {}

  ObjectWithDict::ObjectWithDict(std::type_info const& typeID, void* address) : object_(Reflex::Type::ByTypeInfo(typeID), address) {}

  std::string
  ObjectWithDict::name() const {
    return object_.TypeOf().Name();
  }    
    
  std::string
  ObjectWithDict::typeName() const {
    return object_.TypeOf().TypeInfo().name();
  }

  bool
  ObjectWithDict::isPointer() const {
    return object_.TypeOf().IsPointer();
  }

  bool
  ObjectWithDict::isReference() const {
    return object_.TypeOf().IsReference();
  }

  bool
  ObjectWithDict::isTypedef() const {
    return object_.TypeOf().IsTypedef();
  }

  TypeWithDict
  ObjectWithDict::typeOf() const {
    return TypeWithDict(object_.TypeOf());
  }

  TypeWithDict
  ObjectWithDict::toType() const {
    return TypeWithDict(object_.TypeOf().ToType());
  }

  TypeWithDict
  ObjectWithDict::finalType() const {
    return TypeWithDict(object_.TypeOf().FinalType());
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

}
