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

    std::string name() const {return function_.Name();}

    std::string typeName() const;

    TypeWithDict declaringType() const;

    TypeWithDict typeOf() const;

    bool isConst() const {
      return function_.IsConst();
    }

    bool isConstructor() const {
      return function_.IsConstructor();
    }

    bool isDestructor() const {
      return function_.IsDestructor();
    }

    bool isOperator() const {
      return function_.IsOperator();
    }

    bool isPublic() const {
      return function_.IsPublic();
    }

    bool isStatic() const {
      return function_.IsStatic();
    }

    TypeWithDict returnType() const;

    size_t functionParameterSize(bool required = false) const {
      return function_.FunctionParameterSize(required);
    }

    void invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values = std::vector<void*>()) const;

    Reflex::Type_Iterator begin() const;
    Reflex::Type_Iterator end() const;
    size_t size() const {
      return functionParameterSize();
    }

#ifndef __GCCXML__
    explicit operator bool() const {
      return bool(function_);
    }
#endif

  private:

    Reflex::Member function_;
  };

}
#endif
