
//#include "Reflex/Object.h"
//#include "Reflex/Member.h"
//#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {

  FunctionWithDict::FunctionWithDict(){}

  FunctionWithDict::FunctionWithDict(Reflex::Member const& func) {}

  Reflex::Member const&
  FunctionWithDict::function() const {
    return *function_;
  } 

  std::string
  FunctionWithDict::name() const {
    //return function().Name();
    return std::string();
  }

  std::string
  FunctionWithDict::typeName() const {
    //return function().TypeOf().Name();
    return std::string();
  }

  TypeWithDict
  FunctionWithDict::typeOf() const {
    //return (TypeWithDict(function().TypeOf()));
    return (TypeWithDict());
  }

  TypeWithDict
  FunctionWithDict::returnType() const {
    //return (TypeWithDict(function().TypeOf().ReturnType()));
    return (TypeWithDict());
  }

  TypeWithDict
  FunctionWithDict::finalReturnType() const {
    //return (TypeWithDict(function().TypeOf().ReturnType().FinalType()));
    return (TypeWithDict());
  }

  TypeWithDict
  FunctionWithDict::declaringType() const {
    //return (TypeWithDict(function().DeclaringType()));
    return (TypeWithDict());
  }

  bool
  FunctionWithDict::isConst() const {
    //return function().IsConst();
    return false;
  }

  bool
  FunctionWithDict::isConstructor() const {
    //return function().IsConstructor();
    return false;
  }

  bool
  FunctionWithDict::isDestructor() const {
    //return function().IsDestructor();
    return false;
  }

  bool
  FunctionWithDict::isOperator() const {
    //return function().IsOperator();
    return false;
  }

  bool
  FunctionWithDict::isPublic() const {
    //return function().IsPublic();
    return false;
  }

  bool FunctionWithDict::isStatic() const {
    //return function().IsStatic();
    return false;
  }

  size_t
  FunctionWithDict::functionParameterSize(bool required) const {
    //return function().FunctionParameterSize(required);
    return 0U;
  }

  void
  FunctionWithDict::invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values) const {
    //Reflex::Object reflexReturn(ret->typeOf().type_, ret->address());
    //function().Invoke(Reflex::Object(obj.typeOf().type_, obj.address()), &reflexReturn, values);
  }

/*
  Reflex::Type_Iterator
  FunctionWithDict::begin() const {
    return function().TypeOf().FunctionParameter_Begin();
  }

  Reflex::Type_Iterator
  FunctionWithDict::end() const {
    return function().TypeOf().FunctionParameter_End();
  }
*/

  FunctionWithDict::operator bool() const {
    return false; 
  }

}
