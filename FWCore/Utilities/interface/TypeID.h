#ifndef FWCore_Utilities_TypeID_h
#define FWCore_Utilities_TypeID_h

/*----------------------------------------------------------------------
  
TypeID: A unique identifier for a C++ type.

The identifier is unique within an entire program, but can not be
persisted across invocations of the program.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <typeinfo>
#include <string>
#include "FWCore/Utilities/interface/TypeIDBase.h"

namespace edm {

  class TypeID : private TypeIDBase {
  public:

    TypeID() : TypeIDBase() {}

    explicit TypeID(std::type_info const& t) : TypeIDBase(t) {
    }

    template <typename T>
    explicit TypeID(T const& t) : TypeIDBase(typeid(t)) {
    }

    static TypeID
    byName(std::string const& className);

    // Print out the name of the type, using the dictionary class name.
    void print(std::ostream& os) const;

    std::string className() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

    bool hasDictionary() const;

    bool isComplete() const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif
    
    using TypeIDBase::name;

    bool operator<(TypeID const& b) const { return this->TypeIDBase::operator<(b); }

    bool operator==(TypeID const& b) const {return this->TypeIDBase::operator==(b);}

    using TypeIDBase::typeInfo;

  private:
    static bool stripTemplate(std::string& theName);

    static bool stripNamespace(std::string& theName);

  };

  inline bool operator>(TypeID const& a, TypeID const& b) {
    return b < a;
  }

  inline bool operator!=(TypeID const& a, TypeID const& b) {
    return !(a == b);
  }

  std::ostream& operator<<(std::ostream& os, TypeID const& id);
}
#endif
