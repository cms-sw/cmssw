#ifndef FWCore_Utilities_ObjectWithDict_h
#define FWCore_Utilities_ObjectWithDict_h

/*----------------------------------------------------------------------
  
ObjectWithDict:  A holder for an object and its type information.

----------------------------------------------------------------------*/
#include <string>
#include <typeinfo>

#include "Reflex/Object.h"

namespace edm {
  class TypeWithDict;

  class ObjectWithDict {
  public:
    ObjectWithDict() : object_() {}

    explicit ObjectWithDict(TypeWithDict const& type);

    ObjectWithDict(TypeWithDict const& type,
                   TypeWithDict const& signature,
                   std::vector<void*> const& values);



    ObjectWithDict(TypeWithDict const& type, void* address);

    ObjectWithDict(std::type_info const& typeID, void* address);

    void destruct() const {
      object_.Destruct();
    }

    void* address() const {
      return object_.Address();
    }

    std::string typeName() const;

    bool isPointer() const;

    bool isReference() const;

    bool isTypedef() const;

    TypeWithDict typeOf() const;

    TypeWithDict toType() const;

    TypeWithDict finalType() const;

    TypeWithDict dynamicType() const;

    void invoke(std::string const& fm, ObjectWithDict* ret) const{
      object_.Invoke(fm, &ret->object_);
    }

    ObjectWithDict castObject(TypeWithDict const& type) const;

    ObjectWithDict get(std::string const& member) const {
      return ObjectWithDict(object_.Get(member));
    }

#ifndef __GCCXML__
    explicit operator bool() const {
      return bool(object_);
    }
#endif

    ObjectWithDict construct() const;

    template <typename T> T objectCast() {
      return Reflex::Object_Cast<T>(this->object_);
    }

  private:
    friend class FunctionWithDict;
    friend class MemberWithDict;
    friend class TypeWithDict;

    explicit ObjectWithDict(Reflex::Object const& obj) : object_(obj) {}

    Reflex::Object object_;
  };

}
#endif
