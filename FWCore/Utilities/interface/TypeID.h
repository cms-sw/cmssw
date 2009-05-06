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

  class TypeID : public TypeIDBase {
  public:

    TypeID() : TypeIDBase() {}

    TypeID(TypeID const& other) : TypeIDBase(other) {
    }
    
    explicit TypeID(std::type_info const& t) : TypeIDBase(t) {
    }

    template <typename T>
    explicit TypeID(T const& t) : TypeIDBase(typeid(t)) {
    }

    // Print out the name of the type, using the reflection class name.
    void print(std::ostream& os) const;

    std::string className() const;

    std::string userClassName() const;

    std::string friendlyClassName() const;

    bool hasDictionary() const;

  private:
    static bool stripTemplate(std::string& theName);

    static bool stripNamespace(std::string& theName);

  };

  std::ostream& operator<<(std::ostream& os, TypeID const& id);
}
#endif
