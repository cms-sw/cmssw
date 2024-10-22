#include "CommonTools/Utils/interface/parser/MethodInvoker.h"

#include "CommonTools/Utils/src/ExpressionVar.h"
#include "CommonTools/Utils/interface/parser/MethodSetter.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/interface/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
//#include <iostream>
using namespace reco::parser;
using namespace std;

MethodInvoker::MethodInvoker(const edm::FunctionWithDict& method, const vector<AnyMethodArgument>& ints)
    : method_(method), member_(), ints_(ints), isFunction_(true) {
  setArgs();
  if (isFunction_) {
    retTypeFinal_ = method_.finalReturnType();
  }
  //std::cout <<
  //   "Booking " <<
  //   methodName() <<
  //   " from " <<
  //   method_.declaringType().name() <<
  //   " with " <<
  //   args_.size() <<
  //   " arguments" <<
  //   " (were " <<
  //   ints.size() <<
  //   ")" <<
  //   std::endl;
}

MethodInvoker::MethodInvoker(const edm::MemberWithDict& member)
    : method_(), member_(member), ints_(), isFunction_(false) {
  setArgs();
  //std::cout <<
  //  "Booking " <<
  //  methodName() <<
  //  " from " <<
  //  member_.declaringType().name() <<
  //  " with " <<
  //  args_.size() <<
  //  " arguments" <<
  //  " (were " <<
  //  ints.size() <<
  //  ")" <<
  //  std::endl;
}

MethodInvoker::MethodInvoker(const MethodInvoker& rhs)
    : method_(rhs.method_),
      member_(rhs.member_),
      ints_(rhs.ints_),
      isFunction_(rhs.isFunction_),
      retTypeFinal_(rhs.retTypeFinal_) {
  setArgs();
}

MethodInvoker& MethodInvoker::operator=(const MethodInvoker& rhs) {
  if (this != &rhs) {
    method_ = rhs.method_;
    member_ = rhs.member_;
    ints_ = rhs.ints_;
    isFunction_ = rhs.isFunction_;
    retTypeFinal_ = rhs.retTypeFinal_;

    setArgs();
  }
  return *this;
}

void MethodInvoker::setArgs() {
  for (size_t i = 0; i < ints_.size(); ++i) {
    args_.push_back(std::visit(AnyMethodArgument2VoidPtr(), ints_[i]));
  }
}

std::string MethodInvoker::methodName() const {
  if (isFunction_) {
    return method_.name();
  }
  return member_.name();
}

std::string MethodInvoker::returnTypeName() const {
  if (isFunction_) {
    return method_.typeName();
  }
  return member_.typeOf().qualifiedName();
}

edm::ObjectWithDict MethodInvoker::invoke(const edm::ObjectWithDict& o, edm::ObjectWithDict& retstore) const {
  edm::ObjectWithDict ret = retstore;
  edm::TypeWithDict retType;
  if (isFunction_) {
    //std::cout << "Invoking " << methodName()
    //  << " from " << method_.declaringType().qualifiedName()
    //  << " on an instance of " << o.dynamicType().qualifiedName()
    //  << " at " << o.address()
    //  << " with " << args_.size() << " arguments"
    //  << std::endl;
    method_.invoke(o, &ret, args_);
    // this is correct, it takes pointers and refs into account
    retType = retTypeFinal_;
  } else {
    //std::cout << "Invoking " << methodName()
    //  << " from " << member_.declaringType().qualifiedName()
    //  << " on an instance of " << o.dynamicType().qualifiedName()
    //  << " at " << o.address()
    //  << " with " << args_.size() << " arguments"
    //  << std::endl;
    ret = member_.get(o);
    retType = member_.typeOf();
  }
  void* addr = ret.address();
  //std::cout << "Stored result of " <<  methodName() << " (type " <<
  //  returnTypeName() << ") at " << addr << std::endl;
  if (addr == nullptr) {
    throw edm::Exception(edm::errors::InvalidReference)
        << "method \"" << methodName() << "\" called with " << args_.size() << " arguments returned a null pointer ";
  }
  //std::cout << "Return type is " << retType.qualifiedName() << std::endl;
  if (retType.isPointer() || retType.isReference()) {
    // both need void** -> void* conversion
    if (retType.isPointer()) {
      retType = retType.toType();
    } else {
      // strip cv & ref flags
      // FIXME: This is only true if the propery passed to the constructor
      //       overrides the const and reference flags.
      retType = retType.stripConstRef();
    }
    ret = edm::ObjectWithDict(retType, *static_cast<void**>(addr));
    //std::cout << "Now type is " << retType.qualifiedName() << std::endl;
  }
  if (!bool(ret)) {
    throw edm::Exception(edm::errors::Configuration)
        << "method \"" << methodName() << "\" returned void invoked on object of type \"" << o.typeOf().qualifiedName()
        << "\"\n";
  }
  return ret;
}

LazyInvoker::LazyInvoker(const std::string& name, const std::vector<AnyMethodArgument>& args)
    : name_(name), argsBeforeFixups_(args) {}

LazyInvoker::~LazyInvoker() {}

const SingleInvoker& LazyInvoker::invoker(const edm::TypeWithDict& type) const {
  //std::cout << "LazyInvoker for " << name_ << " called on type " <<
  //  type.qualifiedName() << std::endl;
  const edm::TypeID thetype(type.typeInfo());
  auto found = invokers_.find(thetype);
  if (found != invokers_.cend()) {
    return *(found->second);
  }
  auto to_add = std::make_shared<SingleInvoker>(type, name_, argsBeforeFixups_);
  auto emplace_result = invokers_.insert(std::make_pair(thetype, to_add));
  return *(emplace_result.first->second);
}

edm::ObjectWithDict LazyInvoker::invoke(const edm::ObjectWithDict& o, std::vector<StorageManager>& v) const {
  pair<edm::ObjectWithDict, bool> ret(o, false);
  do {
    edm::TypeWithDict type = ret.first.typeOf();
    if (type.isClass()) {
      type = ret.first.dynamicType();
    }
    ret = invoker(type).invoke(edm::ObjectWithDict(type, ret.first.address()), v);
  } while (ret.second == false);
  return ret.first;
}

double LazyInvoker::invokeLast(const edm::ObjectWithDict& o, std::vector<StorageManager>& v) const {
  pair<edm::ObjectWithDict, bool> ret(o, false);
  const SingleInvoker* i = nullptr;
  do {
    edm::TypeWithDict type = ret.first.typeOf();
    if (type.isClass()) {
      type = ret.first.dynamicType();
    }
    i = &invoker(type);
    ret = i->invoke(edm::ObjectWithDict(type, ret.first.address()), v);
  } while (ret.second == false);
  return i->retToDouble(ret.first);
}

SingleInvoker::SingleInvoker(const edm::TypeWithDict& type,
                             const std::string& name,
                             const std::vector<AnyMethodArgument>& args) {
  TypeStack typeStack(1, type);
  LazyMethodStack dummy;
  MethodArgumentStack dummy2;
  MethodSetter setter(invokers_, dummy, typeStack, dummy2, false);
  isRefGet_ = !setter.push(name, args, "LazyInvoker dynamic resolution", false);
  //std::cerr  << "SingleInvoker on type " <<  type.qualifiedName() <<
  //  ", name " << name << (isRefGet_ ? " is just a ref.get " : " is real") <<
  //  std::endl;
  returnStorage(createStorage(storageNeedsDestructor_));
  // typeStack[0] = type of self
  // typeStack[1] = type of ret
  retType_ = reco::typeCode(typeStack[1]);
}

SingleInvoker::~SingleInvoker() {
  edm::ObjectWithDict stored;
  while (storage_.try_pop(stored)) {
    //std::cout <<"deleting "<<stored.address()<<" from "<<this<<std::endl;
    ExpressionVar::delStorage(stored);
  }
}

edm::ObjectWithDict SingleInvoker::createStorage(bool& needsDestructor) const {
  if (invokers_.front().isFunction()) {
    edm::TypeWithDict retType = invokers_.front().method().finalReturnType();
    edm::ObjectWithDict stored;
    needsDestructor = ExpressionVar::makeStorage(stored, retType);
    return stored;
  }
  needsDestructor = false;
  return edm::ObjectWithDict();
}

edm::ObjectWithDict SingleInvoker::borrowStorage() const {
  edm::ObjectWithDict o;
  if (storage_.try_pop(o)) {
    //std::cout <<"borrowed "<<o.address()<<" from "<<this<<std::endl;
    return o;
  }
  bool dummy;
  o = createStorage(dummy);
  //std::cout <<"borrowed new "<<o.address()<<std::endl;
  return o;
}

void SingleInvoker::returnStorage(edm::ObjectWithDict&& o) const {
  //std::cout <<"returned "<<o.address()<<" to "<<this<<std::endl;
  storage_.push(std::move(o));
}

pair<edm::ObjectWithDict, bool> SingleInvoker::invoke(const edm::ObjectWithDict& o,
                                                      std::vector<StorageManager>& v) const {
  // std::cerr << "[SingleInvoker::invoke] member " <<
  //   invokers_.front().method().qualifiedName() <<
  //   " of type " <<
  //   o.typeOf().qualifiedName() <<
  //   (!isRefGet_ ? " is one shot" : " needs another round") <<
  //   std::endl;
  auto storage = borrowStorage();
  pair<edm::ObjectWithDict, bool> ret(invokers_.front().invoke(o, storage), !isRefGet_);
  v.emplace_back(storage, this, storageNeedsDestructor_);
  return ret;
}

double SingleInvoker::retToDouble(const edm::ObjectWithDict& o) const {
  if (!ExpressionVar::isValidReturnType(retType_)) {
    throwFailedConversion(o);
  }
  return ExpressionVar::objToDouble(o, retType_);
}

void SingleInvoker::throwFailedConversion(const edm::ObjectWithDict& o) const {
  throw edm::Exception(edm::errors::Configuration)
      << "member \"" << invokers_.back().methodName() << "\" return type is \"" << invokers_.back().returnTypeName()
      << "\" retured a \"" << o.typeOf().qualifiedName() << "\" which is not convertible to double.";
}

StorageManager::~StorageManager() {
  if (needsDestructor_) {
    object_.destruct(false);
  }
  if (invoker_) {
    invoker_->returnStorage(std::move(object_));
  }
}
