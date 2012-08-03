#include "CommonTools/Utils/src/MethodInvoker.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/MethodSetter.h"
#include "CommonTools/Utils/src/ExpressionVar.h"

#include <algorithm>
using namespace reco::parser;
using namespace std;

MethodInvoker::MethodInvoker(const edm::MemberWithDict & method, const vector<AnyMethodArgument> & ints) :
  method_(method), ints_(ints), isFunction_(method.isFunctionMember())
{ 
  setArgs();
  /*std::cout << "Booking " << method_.name() 
            << " from " << method_.declaringTy[e().name() 
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

edm::ObjectWithDict
MethodInvoker::invoke(const edm::ObjectWithDict & o, edm::ObjectWithDict &retstore) const {
  edm::ObjectWithDict ret = retstore;
  /*std::cout << "Invoking " << method_.name() 
            << " from " << method_.declaringTy[e().name(edm::TypeNameHandling::Qualified) 
            << " on an instance of " << o.dynamicType().name(edm::TypeNameHandling::Qualified) 
            << " at " << o.address()
            << " with " << args_.size() << " arguments"
            << std::endl; */
  edm::TypeWithDict retType;
  if(isFunction_) {
     method_.invoke(o, &ret, args_);
     retType = method_.typeOf().returnType(); // this is correct, it takes pointers and refs into account
  } else {
     ret = method_.get(o);
     retType = method_.typeOf();
  }
  void * addr = ret.address(); 
  //std::cout << "Stored result of " <<  method_.name() << " (type " << method_.typeOf().returnType().name(edm::TypeNameHandling::Qualified) << ") at " << addr << std::endl;
  if(addr==0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << method_.name() << "\" called with " << args_.size() 
      << " arguments returned a null pointer ";   
  //std::cout << "Return type is " << retType.name(edm::TypeNameHandling::Qualified) << std::endl;
   
  if(retType.isPointer() || retType.isReference()) { // both need (void **)->(void *) conversion
      if (retType.isPointer()) {
        retType = retType.toType(); // for Pointers, I get the real type this way
      } else {
        retType = edm::TypeWithDict(retType, edm::TypeModifiers::NoMod); // strip cv & ref flags
      }
      while (retType.isTypedef()) retType = retType.toType();
      ret = edm::ObjectWithDict(retType, *static_cast<void **>(addr));
      //std::cout << "Now type is " << retType.name(edm::TypeNameHandling::Qualified) << std::endl;
  }
  if(!ret) 
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << method_.name() 
      << "\" returned void invoked on object of type \"" 
      << o.typeOf().name(edm::TypeNameHandling::Qualified) << "\"\n";
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
    //std::cout << "LazyInvoker for " << name_ << " called on type " << type.name(edm::TypeNameHandling::Qualified) << std::endl;
    SingleInvokerPtr & invoker = invokers_[type.id()];
    if (!invoker) {
        //std::cout << "  Making new invoker for " << name_ << " on type " << type.name(edm::TypeNameHandling::Qualified) << std::endl;
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
    storageNeedsDestructor_ = ExpressionVar::makeStorage(storage_, invokers_.front().method());
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
        << "member \"" << invokers_.back().method().name()
        << "\" return type is \"" << invokers_.back().method().typeOf().name(edm::TypeNameHandling::Qualified)
        << "\" retured a \"" << o.typeOf().name(edm::TypeNameHandling::Qualified)
        << "\" which is not convertible to double.";
}
