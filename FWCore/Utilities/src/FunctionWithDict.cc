#include "FWCore/Utilities/interface/FunctionWithDict.h"

#include "FWCore/Utilities/interface/IterWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TInterpreter.h"
#include "TMethod.h"
#include "TMethodArg.h"

namespace edm {

  FunctionWithDict::FunctionWithDict() : function_(nullptr) {
  }

  FunctionWithDict::FunctionWithDict(TMethod* meth) : function_(meth) {
  }

  FunctionWithDict::operator bool() const {
    if (function_ == nullptr) {
      return false;
    }
    return function_->IsValid();
  }

  std::string FunctionWithDict::name() const {
    return function_->GetName();
  }

  std::string
  FunctionWithDict::typeName() const {
    return function_->GetReturnTypeName();
  }

  TypeWithDict
  FunctionWithDict::typeOf() const {
    return TypeWithDict::byName(function_->GetReturnTypeName());
  }

  TypeWithDict
  FunctionWithDict::returnType() const {
    return TypeWithDict::byName(function_->GetReturnTypeName());
  }

  TypeWithDict
  FunctionWithDict::finalReturnType() const {
    return TypeWithDict::byName(function_->GetReturnTypeNormalizedName());
  }

  TypeWithDict
  FunctionWithDict::declaringType() const {
    return TypeWithDict(function_->GetClass());
  }

  bool
  FunctionWithDict::isConst() const {
    return function_->Property() & kIsConstMethod;
  }

  bool
  FunctionWithDict::isConstructor() const {
    return function_->ExtraProperty() & kIsConstructor;
  }

  bool
  FunctionWithDict::isDestructor() const {
    return function_->ExtraProperty() & kIsDestructor;
  }

  bool
  FunctionWithDict::isOperator() const {
    return function_->ExtraProperty() & kIsOperator;
  }

  bool
  FunctionWithDict::isPublic() const {
    return function_->Property() & kIsPublic;
  }

  bool FunctionWithDict::isStatic() const {
    return function_->Property() & kIsStatic;
  }

  size_t
  FunctionWithDict::functionParameterSize(bool required/*= false*/) const {
    if (required) {
      return function_->GetNargs() - function_->GetNargsOpt();
    }
    return function_->GetNargs();
  }

  size_t
  FunctionWithDict::size() const {
    return function_->GetNargs();
  }

  /// Call a member function.
  void
  FunctionWithDict::invoke(ObjectWithDict const& obj, ObjectWithDict* ret/*=nullptr*/,
         std::vector<void*> const& values/*=std::vector<void*>()*/) const {
    //Reflex::Object reflexReturn(ret->typeOf().type_, ret->address());
    //function_.Invoke(Reflex::Object(obj.typeOf().type_, obj.address()), &reflexReturn, values);
    if (ret == nullptr) {
      gInterpreter->ExecuteWithArgsAndReturn(function_, obj.address(), values, 0);
      return;
    }
    gInterpreter->ExecuteWithArgsAndReturn(function_, obj.address(), values, ret->address());
  }

  /// Call a static function.
  void
  FunctionWithDict::invoke(ObjectWithDict* ret/*=nullptr*/,
         std::vector<void*> const& values/*=std::vector<void*>()*/) const {
    //Reflex::Object reflexReturn(ret->typeOf().type_, ret->address());
    //function_.Invoke(obj.address()), &reflexReturn, values);
    if (ret == nullptr) {
      gInterpreter->ExecuteWithArgsAndReturn(function_, 0, values, 0);
      return;
    }
    gInterpreter->ExecuteWithArgsAndReturn(function_, 0, values, ret->address());
  }

  IterWithDict<TMethodArg>
  FunctionWithDict::begin() const {
    if (function_ == nullptr) {
      return IterWithDict<TMethodArg>();
    }
    return IterWithDict<TMethodArg>(function_->GetListOfMethodArgs());
  }

  IterWithDict<TMethodArg>
  FunctionWithDict::end() const {
    return IterWithDict<TMethodArg>();
  }

} // namespace edm
