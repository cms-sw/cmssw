#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>
using namespace reco::parser;
using namespace ROOT::Reflex;
using namespace std;

MethodInvoker::MethodInvoker(const Member & method, const vector<int> & ints) :
  method_(method), ints_(ints) { 
  setArgs();
}

MethodInvoker::MethodInvoker(const MethodInvoker & other) :
  method_(other.method_), ints_(other.ints_) {
  setArgs();
}

MethodInvoker & MethodInvoker::operator=(const MethodInvoker & other) {
  method_ = other.method_;
  ints_ = other.ints_;
  setArgs();
  return *this;
}

void MethodInvoker::setArgs() {
 for(size_t i = 0; i < ints_.size(); ++i) {
    args_.push_back((void *)(&ints_[i]));
  }
}

Object MethodInvoker::value(const Object & o) const {
  Object ret;
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
  ret = method_.Invoke(o, args_);
  void * addr = ret.Address(); 
  if(addr==0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << method_.Name() << "\" called with " << args_.size() 
      << " arguments returned a null pointer ";   
  Type retType = ret.TypeOf();
  bool stripped = false;
  while(retType.IsTypedef()) { 
    retType = retType.ToType(); stripped = true; 
  }
  bool isPtr = retType.IsPointer() /* isRef = retType.IsReference() */;
  if(isPtr) {
    if(!stripped) {
      stripped = true;
      retType = retType.ToType();
      while(retType.IsTypedef()) {
	retType = retType.ToType();
      }
    }
  }
  if(stripped)
    ret = Object(retType, addr);
  if(!ret) 
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << method_.Name() 
      << "\" returned void invoked on object of type " 
      << o.TypeOf().Name() << "\n";
  return ret;
}
