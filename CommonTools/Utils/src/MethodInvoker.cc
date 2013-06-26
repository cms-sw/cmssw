#include "CommonTools/Utils/src/MethodInvoker.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/MethodSetter.h"
#include "CommonTools/Utils/src/ExpressionVar.h"

#include <algorithm>
using namespace reco::parser;
using namespace std;

MethodInvoker::MethodInvoker(const edm::FunctionWithDict & method, const vector<AnyMethodArgument> & ints) :
  method_(method), member_(), ints_(ints), isFunction_(true)
{ 
  setArgs();
  /*std::cout << "Booking " << methodName() 
            << " from " << method_.declaringType().name() 
            << " with " << args_.size() << " arguments"
            << " (were " << ints.size() << ")"
            << std::endl;*/
}

MethodInvoker::MethodInvoker(const edm::MemberWithDict & member) :
  method_(), member_(member), ints_(), isFunction_(false)
{ 
  setArgs();
  /*std::cout << "Booking " << methodName() 
            << " from " << member_.declaringType().name() 
            << " with " << args_.size() << " arguments"
            << " (were " << ints.size() << ")"
            << std::endl;*/
}

MethodInvoker::MethodInvoker(const MethodInvoker & other) :
  method_(other.method_), member_(other.member_), ints_(other.ints_), isFunction_(other.isFunction_) {
  setArgs();
}

MethodInvoker & MethodInvoker::operator=(const MethodInvoker & other) {
  method_ = other.method_;
  member_ = other.member_;
  ints_ = other.ints_;
  isFunction_ = other.isFunction_;
  setArgs();
  return *this;
}

void MethodInvoker::setArgs() {
  for(size_t i = 0; i < ints_.size(); ++i) {
      args_.push_back( boost::apply_visitor( AnyMethodArgument2VoidPtr(), ints_[i] ) );
  }
}

std::string
MethodInvoker::methodName() const {
  if(isFunction_) {
     return method_.name();
  }
  return member_.name();
}

std::string
MethodInvoker::returnTypeName() const {
  if(isFunction_) {
     return method_.typeOf().qualifiedName();
  }
  return member_.typeOf().qualifiedName();
}

edm::ObjectWithDict
MethodInvoker::invoke(const edm::ObjectWithDict & o, edm::ObjectWithDict &retstore) const {
  edm::ObjectWithDict ret = retstore;
  edm::TypeWithDict retType;
  if(isFunction_) {
     /*std::cout << "Invoking " << methodName() 
            << " from " << method_.declaringType().qualifiedName() 
            << " on an instance of " << o.dynamicType().qualifiedName() 
            << " at " << o.address()
            << " with " << args_.size() << " arguments"
            << std::endl; */
     method_.invoke(o, &ret, args_);
     retType = method_.finalReturnType(); // this is correct, it takes pointers and refs into account
  } else {
     /*std::cout << "Invoking " << methodName() 
            << " from " << member_.declaringType().qualifiedName() 
            << " on an instance of " << o.dynamicType().qualifiedName() 
            << " at " << o.address()
            << " with " << args_.size() << " arguments"
            << std::endl; */
     ret = member_.get(o);
     retType = member_.typeOf();
  }
  void * addr = ret.address(); 
  //std::cout << "Stored result of " <<  methodName() << " (type " << returnTypeName() << ") at " << addr << std::endl;
  if(addr==0) {
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << methodName() << "\" called with " << args_.size() 
      << " arguments returned a null pointer ";   
  }
  //std::cout << "Return type is " << retType.qualifiedName() << std::endl;
   
  if(retType.isPointer() || retType.isReference()) { // both need (void **)->(void *) conversion
      if (retType.isPointer()) {
        retType = retType.toType(); // for Pointers, I get the real type this way
      } else {
        retType = edm::TypeWithDict(retType, 0L); // strip cv & ref flags
      }
      ret = edm::ObjectWithDict(retType, *static_cast<void **>(addr));
      //std::cout << "Now type is " << retType.qualifiedName() << std::endl;
  }
  if(!ret) {
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << methodName()
      << "\" returned void invoked on object of type \"" 
      << o.typeOf().qualifiedName() << "\"\n";
  }
  return ret;
}

LazyInvoker::LazyInvoker(const std::string &name, const std::vector<AnyMethodArgument> & args) :
    name_(name),
    argsBeforeFixups_(args)
{
}

LazyInvoker::~LazyInvoker() 
{
}

const SingleInvoker &
LazyInvoker::invoker(const edm::TypeWithDict & type) const 
{
    //std::cout << "LazyInvoker for " << name_ << " called on type " << type.qualifiedName() << std::endl;
    SingleInvokerPtr & invoker = invokers_[edm::TypeID(type.id())];
    if (!invoker) {
        //std::cout << "  Making new invoker for " << name_ << " on type " << type.qualifiedName() << std::endl;
        invoker.reset(new SingleInvoker(type, name_, argsBeforeFixups_));
    } 
    return * invoker;
}

edm::ObjectWithDict
LazyInvoker::invoke(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const 
{
    pair<edm::ObjectWithDict, bool> ret(o,false);
    do {    
        edm::TypeWithDict type = ret.first.typeOf();
        if (type.isClass()) type = ret.first.dynamicType();
        ret = invoker(type).invoke(edm::ObjectWithDict(type, ret.first.address()), v);
    } while (ret.second == false);
    return ret.first; 
}

double
LazyInvoker::invokeLast(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const 
{
    pair<edm::ObjectWithDict, bool> ret(o,false);
    const SingleInvoker *i = 0;
    do {    
        edm::TypeWithDict type = ret.first.typeOf();
        if (type.isClass()) type = ret.first.dynamicType();
        i = & invoker(type);
        ret = i->invoke(edm::ObjectWithDict(type, ret.first.address()), v);
    } while (ret.second == false);
    return i->retToDouble(ret.first);
}

SingleInvoker::SingleInvoker(const edm::TypeWithDict &type,
        const std::string &name,
        const std::vector<AnyMethodArgument> &args) 
{
    TypeStack typeStack(1, type);
    LazyMethodStack dummy;
    MethodArgumentStack dummy2;
    MethodSetter setter(invokers_, dummy, typeStack, dummy2, false);
    isRefGet_ = !setter.push(name, args, "LazyInvoker dynamic resolution", false);
    //std::cerr  << "SingleInvoker on type " <<  type.qualifiedName() << ", name " << name << (isRefGet_ ? " is just a ref.get " : " is real") << std::endl;
    if(invokers_.front().isFunction()) {
       edm::TypeWithDict retType = invokers_.front().method().finalReturnType();
       storageNeedsDestructor_ = ExpressionVar::makeStorage(storage_, retType);
    } else {
       storage_ = edm::ObjectWithDict();
       storageNeedsDestructor_ = false;
    }
    retType_ = reco::typeCode(typeStack[1]); // typeStack[0] = type of self, typeStack[1] = type of ret
}

SingleInvoker::~SingleInvoker()
{
    ExpressionVar::delStorage(storage_);
}

pair<edm::ObjectWithDict,bool>
SingleInvoker::invoke(const edm::ObjectWithDict & o, std::vector<edm::ObjectWithDict> &v) const 
{
    /* std::cerr << "[SingleInvoker::invoke] member " << invokers_.front().method().qualifiedName() << 
                                       " of type " << o.typeOf().qualifiedName() <<
                                       (!isRefGet_ ? " is one shot" : " needs another round") << std::endl; */
    pair<edm::ObjectWithDict,bool> ret(invokers_.front().invoke(o, storage_), !isRefGet_);
    if (storageNeedsDestructor_) {
        //std::cout << "Storage type: " << storage_.typeOf().qualifiedName() << ", I have to call the destructor." << std::endl;
        v.push_back(storage_);
    }
    return ret;
}

double
SingleInvoker::retToDouble(const edm::ObjectWithDict & o) const {
    if (!ExpressionVar::isValidReturnType(retType_)) {
        throwFailedConversion(o);
    }
    return ExpressionVar::objToDouble(o, retType_);
}

void
SingleInvoker::throwFailedConversion(const edm::ObjectWithDict & o) const {
    throw edm::Exception(edm::errors::Configuration)
        << "member \"" << invokers_.back().methodName()
        << "\" return type is \"" << invokers_.back().returnTypeName()
        << "\" retured a \"" << o.typeOf().qualifiedName()
        << "\" which is not convertible to double.";
}
