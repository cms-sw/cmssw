#ifndef FWCore_Reflection_ObjectWithDict_h
#define FWCore_Reflection_ObjectWithDict_h

/*----------------------------------------------------------------------

ObjectWithDict:  A holder for an object and its type information.

----------------------------------------------------------------------*/

#include <string>
#include <typeinfo>

#include "FWCore/Reflection/interface/TypeWithDict.h"

namespace edm {

  class ObjectWithDict {
  private:
    TypeWithDict type_;
    void* address_;

  public:
    static ObjectWithDict byType(TypeWithDict const&);

  public:
    ObjectWithDict() : address_(nullptr) {}
    explicit ObjectWithDict(TypeWithDict const& type, void* address) : type_(type), address_(address) {}
    explicit ObjectWithDict(std::type_info const& type, void* address) : type_(TypeWithDict(type)), address_(address) {}
    explicit operator bool() const { return bool(type_) && (address_ != nullptr); }
    void* address() const { return address_; }
    TypeWithDict const& typeOf() const { return type_; }
    TypeWithDict dynamicType() const;
    ObjectWithDict castObject(TypeWithDict const&) const;
    ObjectWithDict get(std::string const& memberName) const;
    //ObjectWithDict construct() const;
    void destruct(bool dealloc) const;

    template <typename T>
    T& objectCast() {
      return *reinterpret_cast<T*>(address());
    }

    template <typename T>
    T const& objectCast() const {
      return *reinterpret_cast<T*>(address());
    }
  };

}  // namespace edm

#endif  // FWCore_Reflection_ObjectWithDict_h
