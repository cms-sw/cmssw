#include "FWCore/Utilities/interface/FunctionWithDict.h"

#include "FWCore/Utilities/interface/IterWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TMethod.h"
#include "TMethodArg.h"
#include "TMethodCall.h"

#include "tbb/concurrent_unordered_map.h"


namespace edm {
  namespace {
    typedef tbb::concurrent_unordered_map<TMethod const*, TypeWithDict> Map;
    Map returnTypeMap;
  }

  FunctionWithDict::FunctionWithDict() : function_(nullptr) {
  }

  FunctionWithDict::FunctionWithDict(TMethod* meth) : function_(meth) {
    if (meth and isPublic() and not isDestructor() and not isConstructor()) {
      TMethodCall caller( meth );
      auto callFunc = caller.GetCallFunc();
      funcptr_ = gInterpreter->CallFunc_IFacePtr( callFunc );
    }
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
    return function_->GetReturnTypeNormalizedName();
  }

  TypeWithDict
  FunctionWithDict::finalReturnType() const {
    auto const& item = returnTypeMap.find(function_);
    if(item != returnTypeMap.end()) {
       return item->second;
    }
    TypeWithDict theType = TypeWithDict::byName(function_->GetReturnTypeNormalizedName());
    returnTypeMap.insert(std::make_pair(function_, theType));
    return theType;
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
    void ** data = const_cast<void**>(values.data());
    assert(funcptr_.fGeneric);
    if (ret == nullptr) {
      (*funcptr_.fGeneric)(obj.address(), values.size(), data, nullptr);
      return;
    }
    (*funcptr_.fGeneric)(obj.address(), values.size(), data, ret->address());
  }

  /// Call a static function.
  void
  FunctionWithDict::invoke(ObjectWithDict* ret/*=nullptr*/,
         std::vector<void*> const& values/*=std::vector<void*>()*/) const {
    void ** data = const_cast<void **>(values.data());
    assert(funcptr_.fGeneric);
    if (ret == nullptr) {
      (*funcptr_.fGeneric)(nullptr, values.size(), data, nullptr);
      return;
    }
    (*funcptr_.fGeneric)(nullptr, values.size(), data, ret->address());
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
