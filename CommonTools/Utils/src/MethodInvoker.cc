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
            << std::endl; */
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
  /* no need to check the type at run time
  if(method_.IsVirtual()) {
    Type dynType = o.DynamicType();
    Member met = reco::findMethod(dynType, method_.Name(), args_.size()).first;
    if(!met)
      throw edm::Exception(edm::errors::InvalidReference)
	<< "method \"" << method_.Name() << "\" not found in dynamic type \"" 
	<< dynType.Name() << "\"\n";
    ret = met.Invoke(Object(dynType, o.Address()), args_);
    } else */
  /*std::cout << "Invoking " << method_.Name() 
            << " from " << method_.DeclaringType().Name(QUALIFIED) 
            << " on an instance of " << o.DynamicType().Name(QUALIFIED) 
            << " at " << o.Address()
            << " with " << args_.size() << " arguments"
            << std::endl;*/
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
        //invoker = SingleInvokerPtr(new SingleInvoker(type, name_, argsBeforeFixups_));
        invoker.reset(new SingleInvoker(type, name_, argsBeforeFixups_));
    } 
    return * invoker;
}

Object
LazyInvoker::invoke(const Reflex::Object & o) const 
{
    Type type = o.TypeOf();
    if (type.IsClass()) type = o.DynamicType();
    return invoker(type).invoke(Object(type, o.Address()));
}

double
LazyInvoker::invokeLast(const Reflex::Object & o) const 
{
    Type type = o.TypeOf();
    if (type.IsClass()) type = o.DynamicType();
    const SingleInvoker & i = invoker(type);
    Object ret = i.invoke(Object(type, o.Address()));
    return i.retToDouble(ret);
}

SingleInvoker::SingleInvoker(const Reflex::Type &type,
        const std::string &name,
        const std::vector<AnyMethodArgument> &args) 
{
    TypeStack typeStack(1, type);
    LazyMethodStack dummy;
    MethodArgumentStack dummy2;
    MethodSetter setter(invokers_, dummy, typeStack, dummy2, false);
    setter.push(name, args, "LazyInvoker dynamic resolution");
    objects_.resize(invokers_.size());
    std::vector<MethodInvoker>::iterator it, ed; 
    std::vector<Reflex::Object>::iterator ito;
    for (it = invokers_.begin(), ed = invokers_.end(), ito = objects_.begin(); it != ed; ++it, ++ito) {
        ExpressionVar::makeStorage(*ito, it->method());
    }
    retType_ = reco::typeCode(typeStack.back());
}

SingleInvoker::~SingleInvoker()
{
    for (std::vector<Reflex::Object>::iterator it = objects_.begin(), ed = objects_.end(); it != ed; ++it) {
        ExpressionVar::delStorage(*it);
    }
}

Object
SingleInvoker::invoke(const Reflex::Object & o) const 
{
    Object ro = o;
    std::vector<MethodInvoker>::const_iterator itm, end = invokers_.end();
    std::vector<Reflex::Object>::iterator      ito;
    for(itm = invokers_.begin(), ito = objects_.begin(); itm != end; ++itm, ++ito) {
        ro = itm->invoke(ro, *ito);
    }
    return ro;
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
