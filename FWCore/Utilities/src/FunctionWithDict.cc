
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  FunctionWithDict::FunctionWithDict()(new Reflex::Member) {}

  FunctionWithDict::FunctionWithDict(Reflex::Member const& func) : function()(new Reflex::Member(func)) {}

  Reflex Member const&
  FunctionWithDict function() const {
    return *function_;
  } 

  std::string
  FunctionWithDict::name() const {
    return function().Name();
  }

  std::string
  FunctionWithDict::typeName() const {return function().TypeOf().Name();}

  TypeWithDict
  FunctionWithDict::typeOf() const {
    return (TypeWithDict(function().TypeOf()));
  }

  TypeWithDict
  FunctionWithDict::returnType() const {
    return (TypeWithDict(function().TypeOf().ReturnType()));
  }

  TypeWithDict
  FunctionWithDict::finalReturnType() const {
    return (TypeWithDict(function().TypeOf().ReturnType().FinalType()));
  }

  TypeWithDict
  FunctionWithDict::declaringType() const {
    return (TypeWithDict(function().DeclaringType()));
  }

  bool
  FunctionWithDict::isConst() const {
    return function().IsConst();
  }

  bool
  FunctionWithDict::isConstructor() const {
    return function().IsConstructor();
  }

  bool
  FunctionWithDict::isDestructor() const {
    return function().IsDestructor();
  }

  bool
  FunctionWithDict::isOperator() const {
    return function().IsOperator();
  }

  bool
  FunctionWithDict::isPublic() const {
    return function().IsPublic();
  }

  bool FunctionWithDict::isStatic() const {
    return function().IsStatic();
  }

  size_t
  FunctionWithDict::functionParameterSize(bool required) const {
    return function().FunctionParameterSize(required);
  }

  void
  FunctionWithDict::invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values) const {
    Reflex::Object reflexReturn(ret->typeOf().type_, ret->address());
    function().Invoke(Reflex::Object(obj.typeOf().type_, obj.address()), &reflexReturn, values);
  }

  Reflex::Type_Iterator
  FunctionWithDict::begin() const {
    return function().TypeOf().FunctionParameter_Begin();
  }

  Reflex::Type_Iterator
  FunctionWithDict::end() const {
    return function().TypeOf().FunctionParameter_End();
  }

  FunctionWithDict::operator bool() const {
    return bool(function());
  }

}
