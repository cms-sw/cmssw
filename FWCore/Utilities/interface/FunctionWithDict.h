#ifndef FWCore_Utilities_FunctionWithDict_h
#define FWCore_Utilities_FunctionWithDict_h

/*----------------------------------------------------------------------
  
FunctionWithDict:  A holder for a class member function

----------------------------------------------------------------------*/

#include <string>

#include "Reflex/Member.h"

namespace edm {

  class ObjectWithDict;
  class TypeWithDict;

  class FunctionWithDict {
  public:
    FunctionWithDict() : function_() {}

    explicit FunctionWithDict(Reflex::Member const& function) : function_(function) {}

    std::string name() const;

    std::string typeName() const;

    TypeWithDict declaringType() const;

    TypeWithDict typeOf() const;

    bool isConst() const;

    bool isConstructor() const;

    bool isDestructor() const;

    bool isOperator() const;

    bool isPublic() const;

    bool isStatic() const;

    TypeWithDict returnType() const;

    size_t functionParameterSize(bool required = false) const;

    void invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values = std::vector<void*>()) const;

    Reflex::Type_Iterator begin() const;
    Reflex::Type_Iterator end() const;
    size_t size() const {
      return functionParameterSize();
    }

#ifndef __GCCXML__
    explicit operator bool() const;
#endif

  private:

    Reflex::Member function_;
  };

}
#endif
