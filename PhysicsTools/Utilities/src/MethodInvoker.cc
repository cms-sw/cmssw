#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
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
