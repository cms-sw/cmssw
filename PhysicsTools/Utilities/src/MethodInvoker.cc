#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco::parser;
using namespace ROOT::Reflex;
using namespace std;

Object MethodInvoker::value(const Object & o) const {
  Object ret = method_.Invoke(o);
  void * addr = ret.Address(); 
  if(addr==0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << method_.Name() 
      << "\" returned a null pointer ";   
  Type retType = ret.TypeOf();
  bool isRef = retType.IsReference(), isPtr = retType.IsPointer();
  while(retType.IsTypedef()) retType = retType.ToType();
  if(isRef||isPtr) ret = Object(retType, addr);
  if(!ret) 
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << method_.Name() 
      << "\" returned void invoked on object of type " 
      << o.TypeOf().Name() << "\n";
  return ret;
}
