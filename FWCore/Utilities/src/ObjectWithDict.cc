#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <cxxabi.h>

namespace edm {

  ObjectWithDict
  ObjectWithDict::byType(TypeWithDict const& type) {
    ObjectWithDict obj(type.construct());
    return obj;
  }

  ObjectWithDict::ObjectWithDict() : type_(), address_(nullptr) {
  }

  ObjectWithDict::ObjectWithDict(TypeWithDict const& type, void* address) :
    type_(type),
    address_(address) {
  }

  ObjectWithDict::ObjectWithDict(std::type_info const& ti, void* address) :
    type_(TypeWithDict(ti)),
    address_(address) {
  }

  ObjectWithDict::operator bool() const {
    return bool(type_) && (address_ != nullptr);
  }

  void*
  ObjectWithDict::address() const {
    return address_;
  }

  TypeWithDict
  ObjectWithDict::typeOf() const {
    return type_;
  }

  class DummyVT {
  public:
    virtual ~DummyVT();
  };

  DummyVT::~DummyVT() {
  }

  TypeWithDict
  ObjectWithDict::dynamicType() const {
    if (!type_.isVirtual()) {
      return type_;
    }
    // Use a dirty trick, force the typeid() operator
    // to consult the virtual table stored at address_.
    return TypeWithDict(typeid(*(DummyVT*)address_));
  }

  ObjectWithDict
  ObjectWithDict::get(std::string const& memberName) const {
    return type_.dataMemberByName(memberName).get(*this);
  }

  ObjectWithDict
  ObjectWithDict::castObject(TypeWithDict const& to) const {
    TypeWithDict from = typeOf();

    // Same type
    if (from == to) {
      return *this;
    }

    if (to.hasBase(from)) { // down cast
      // use the internal dynamic casting of the compiler (e.g. libstdc++.so)
      void* address = abi::__dynamic_cast(address_, static_cast<abi::__class_type_info const*>(&from.typeInfo()), static_cast<abi::__class_type_info const*>(&to.typeInfo()), -1);
      return ObjectWithDict(to, address);
    }

    if (from.hasBase(to)) { // up cast
      size_t offset = from.getBaseClassOffset(to);
      size_t address = reinterpret_cast<size_t>(address_) + offset;
      return ObjectWithDict(to, reinterpret_cast<void*>(address));
    }

    // if everything fails return the dummy object
    return ObjectWithDict();
  } // castObject

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
    TClass* cl = type_.getClass();
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
