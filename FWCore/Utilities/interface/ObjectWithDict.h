#ifndef FWCore_Utilities_ObjectWithDict_h
#define FWCore_Utilities_ObjectWithDict_h

/*----------------------------------------------------------------------
  
ObjectWithDict:  A holder for an object and its type information.

----------------------------------------------------------------------*/
#include <string>
#include <typeinfo>

#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  class ObjectWithDict {
  public:
    ObjectWithDict();

    ObjectWithDict(TypeWithDict const& type, void* address);

    ObjectWithDict(std::type_info const& typeID, void* address);

    static ObjectWithDict byType(TypeWithDict const& type);

    void* address() const;

    TypeWithDict const& typeOf() const;

    TypeWithDict dynamicType() const;

    ObjectWithDict get(std::string const& memberName) const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif

    template <typename T> T objectCast() {
      return *reinterpret_cast<T*>(address_);
    }

  private:
    friend class FunctionWithDict;
    friend class MemberWithDict;
    friend class TypeWithDict;

    TypeWithDict type_;
    void* address_;
  };

}
#endif
