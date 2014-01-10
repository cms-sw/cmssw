#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  ObjectWithDict
  ObjectWithDict::byType(TypeWithDict const& type) {
    ObjectWithDict obj(type.construct());
    return obj;
  }

  ObjectWithDict::ObjectWithDict() : type_(nullptr), address_(nullptr) {
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type, void* address) :
    type_(gInterpreter->Type_Factory(type.typeInfo())),
    address_(address) {
  }

  ObjectWithDict::ObjectWithDict(std::type_info const& ti, void* address) :
    type_(gInterpreter->Type_Factory(ti)),
    address_(address) {
  }

  ObjectWithDict::operator bool() const {
    return gInterpreter->Type_Bool(type_) && (address_ != nullptr);
  }

  void*
  ObjectWithDict::address() const {
    return address_;
  }

  TypeWithDict
  ObjectWithDict::typeOf() const {
    return TypeWithDict(type_);
  }

  class DummyVT {
  public:
    virtual ~DummyVT();
  };

  DummyVT::~DummyVT() {
  }

  TypeWithDict
  ObjectWithDict::dynamicType() const {
    if (!gInterpreter->Type_IsVirtual(type_)) {
      return TypeWithDict(type_);
    }
    // Use a dirty trick, force the typeid() operator
    // to consult the virtual table stored at address_.
    return TypeWithDict(typeid(*(DummyVT*)address_));
  }

  ObjectWithDict
  ObjectWithDict::get(std::string const& memberName) const {
    return TypeWithDict(type_).dataMemberByName(memberName).get(*this);
  }

  //ObjectWithDict
  //ObjectWithDict::construct() const {
  //  TypeWithDict ty(type_);
  //  TClass* cl = ty.getClass();
  //  if (cl != nullptr) {
  //    return ObjectWithDict(ty, cl->New());
  //  }
  //  return ObjectWithDict(ty, new char[ty.size()]);
  //}

  void
  ObjectWithDict::destruct(bool dealloc) const {
    TypeWithDict ty(type_);
    TClass* cl = ty.getClass();
    if (cl != nullptr) {
      cl->Destructor(address_, !dealloc);
      //if (dealloc) {
      //  address_ = nullptr;
      //}
      return;
    }
    if (dealloc) {
      delete[] reinterpret_cast<char*>(address_);
      //address_ = nullptr;
    }
  }

} // namespace edm
