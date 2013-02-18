#ifndef FWCore_Utilities_ObjectWithDict_h
#define FWCore_Utilities_ObjectWithDict_h

/*----------------------------------------------------------------------
  
ObjectWithDict:  A holder for an object and its type information.

----------------------------------------------------------------------*/
#include <string>
#include <typeinfo>

#include "Reflex/Object.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  class ObjectWithDict {
  public:
    ObjectWithDict();

    ObjectWithDict(TypeWithDict const& type, void* address);

    ObjectWithDict(std::type_info const& typeID, void* address);

    static ObjectWithDict byType(TypeWithDict const& type);

    void destruct() const;

    void* address() const;

    std::string typeName() const;

    bool isPointer() const;

    bool isReference() const;

    bool isTypedef() const;

    TypeWithDict typeOf() const;

    TypeWithDict toType() const;

    TypeWithDict dynamicType() const;

    void invoke(std::string const& fm, ObjectWithDict* ret) const;
    

    ObjectWithDict castObject(TypeWithDict const& type) const;

    ObjectWithDict get(std::string const& member) const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif

    ObjectWithDict construct() const;

    template <typename T> T objectCast() {
      return *reinterpret_cast<T*>(address_);
    }

  private:
    friend class FunctionWithDict;
    friend class MemberWithDict;
    friend class TypeWithDict;

    explicit ObjectWithDict(Reflex::Object const& obj);

    Reflex::Object object_;
    TypeWithDict type_;
    void* address_;
  };

}
#endif
