#include "CommonTools/Utils/src/MethodInvoker.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/MethodSetter.h"
#include "CommonTools/Utils/src/ExpressionVar.h"

#include <algorithm>
using namespace reco::parser;
using namespace Reflex;
using namespace std;

MethodInvoker::MethodInvoker(const Member & method, const vector<AnyMethodArgument> & ints) :
  method_(method), ints_(ints), isFunction_(method.IsFunctionMember())
{ 
  setArgs();
  /*std::cout << "Booking " << method_.Name() 
            << " from " << method_.DeclaringType().Name() 
            << " with " << args_.size() << " arguments"
            << " (were " << ints.size() << ")"
            << std::endl;*/
}

MethodInvoker::MethodInvoker(const MethodInvoker & other) :
  method_(other.method_), ints_(other.ints_), isFunction_(other.isFunction_) {
  setArgs();
}

MethodInvoker & MethodInvoker::operator=(const MethodInvoker & other) {
  method_ = other.method_;
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

Reflex::Object
MethodInvoker::invoke(const Object & o, Reflex::Object &retstore) const {
  Reflex::Object ret = retstore;
  /*std::cout << "Invoking " << method_.Name() 
            << " from " << method_.DeclaringType().Name(QUALIFIED) 
            << " on an instance of " << o.DynamicType().Name(QUALIFIED) 
            << " at " << o.Address()
            << " with " << args_.size() << " arguments"
            << std::endl; */
  Type retType;
  if(isFunction_) {
     method_.Invoke(o, &ret, args_);
     retType = method_.TypeOf().ReturnType(); // this is correct, it takes pointers and refs into account
  } else {
     ret = method_.Get(o);
     retType = method_.TypeOf();
  }
  void * addr = ret.Address(); 
  //std::cout << "Stored result of " <<  method_.Name() << " (type " << method_.TypeOf().ReturnType().Name(QUALIFIED) << ") at " << addr << std::endl;
  if(addr==0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << method_.Name() << "\" called with " << args_.size() 
      << " arguments returned a null pointer ";   
  //std::cout << "Return type is " << retType.Name(QUALIFIED) << std::endl;
   
  if(retType.IsPointer() || retType.IsReference()) { // both need (void **)->(void *) conversion
      if (retType.IsPointer()) {
        retType = retType.ToType(); // for Pointers, I get the real type this way
      } else {
        retType = Type(retType, 0); // strip cv & ref flags
      }
      while (retType.IsTypedef()) retType = retType.ToType();
      ret = Object(retType, *static_cast<void **>(addr));
      //std::cout << "Now type is " << retType.Name(QUALIFIED) << std::endl;
  }
  if(!ret) 
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << method_.Name() 
      << "\" returned void invoked on object of type \"" 
      << o.TypeOf().Name(QUALIFIED) << "\"\n";
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
LazyInvoker::invoker(const Reflex::Type & type) const 
{
    //std::cout << "LazyInvoker for " << name_ << " called on type " << type.Name(QUALIFIED|SCOPED) << std::endl;
    SingleInvokerPtr & invoker = invokers_[type.Id()];
    if (!invoker) {
        //std::cout << "  Making new invoker for " << name_ << " on type " << type.Name(QUALIFIED|SCOPED) << std::endl;
        invoker.reset(new SingleInvoker(type, name_, argsBeforeFixups_));
    } 
    return * invoker;
}

Object
LazyInvoker::invoke(const Reflex::Object & o, std::vector<Reflex::Object> &v) const 
{
    pair<Object, bool> ret(o,false);
    do {    
        Type type = ret.first.TypeOf();
        if (type.IsClass()) type = ret.first.DynamicType();
        ret = invoker(type).invoke(Object(type, ret.first.Address()), v);
    } while (ret.second == false);
    return ret.first; 
}

double
LazyInvoker::invokeLast(const Reflex::Object & o, std::vector<Reflex::Object> &v) const 
{
    pair<Object, bool> ret(o,false);
    const SingleInvoker *i = 0;
    do {    
        Type type = ret.first.TypeOf();
        if (type.IsClass()) type = ret.first.DynamicType();
        i = & invoker(type);
        ret = i->invoke(Object(type, ret.first.Address()), v);
    } while (ret.second == false);
    return i->retToDouble(ret.first);
}

SingleInvoker::SingleInvoker(const Reflex::Type &type,
        const std::string &name,
        const std::vector<AnyMethodArgument> &args) 
{
    TypeStack typeStack(1, type);
    LazyMethodStack dummy;
    MethodArgumentStack dummy2;
    MethodSetter setter(invokers_, dummy, typeStack, dummy2, false);
    isRefGet_ = !setter.push(name, args, "LazyInvoker dynamic resolution", false);
    //std::cerr  << "SingleInvoker on type " <<  type.Name(QUALIFIED|SCOPED) << ", name " << name << (isRefGet_ ? " is just a ref.get " : " is real") << std::endl;
    storageNeedsDestructor_ = ExpressionVar::makeStorage(storage_, invokers_.front().method());
    retType_ = reco::typeCode(typeStack[1]); // typeStack[0] = type of self, typeStack[1] = type of ret
}

SingleInvoker::~SingleInvoker()
{
    ExpressionVar::delStorage(storage_);
}

pair<Object,bool>
SingleInvoker::invoke(const Reflex::Object & o, std::vector<Reflex::Object> &v) const 
{
    /* std::cerr << "[SingleInvoker::invoke] member " << invokers_.front().method().Name(QUALIFIED|SCOPED) << 
                                       " of type " << o.TypeOf().Name(QUALIFIED|SCOPED) <<
                                       (!isRefGet_ ? " is one shot" : " needs another round") << std::endl; */
    pair<Object,bool> ret(invokers_.front().invoke(o, storage_), !isRefGet_);
    if (storageNeedsDestructor_) {
        //std::cout << "Storage type: " << storage_.TypeOf().Name(QUALIFIED|SCOPED) << ", I have to call the destructor." << std::endl;
        v.push_back(storage_);
    }
    return ret;
}

double
SingleInvoker::retToDouble(const Reflex::Object & o) const {
    if (!ExpressionVar::isValidReturnType(retType_)) {
        throwFailedConversion(o);
    }
    return ExpressionVar::objToDouble(o, retType_);
}

void
SingleInvoker::throwFailedConversion(const Reflex::Object & o) const {
    throw edm::Exception(edm::errors::Configuration)
        << "member \"" << invokers_.back().method().Name(QUALIFIED)
        << "\" return type is \"" << invokers_.back().method().TypeOf().Name(QUALIFIED)
        << "\" retured a \"" << o.TypeOf().Name(QUALIFIED)
        << "\" which is not convertible to double.";
}
