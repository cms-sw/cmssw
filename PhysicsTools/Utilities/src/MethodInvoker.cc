#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace reco::parser;
using namespace ROOT::Reflex;
using namespace std;

Object MethodInvoker::value(const Object & o) const {
  cerr << ">>> invoke method of type " << method_.TypeOf().Name() 
       << " on object of type " << o.TypeOf().Name() << endl;
  Object ret = method_.Invoke(o);
  void * addr = ret.Address(); 
  if(addr==0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "method \"" << method_.Name() 
      << "\" returned a null pointer ";   
  Type retType = ret.TypeOf();
  if(retType.IsReference()) {
    cerr << ">>> return type is a reference to " << retType.Name() << endl;
    while(retType.IsTypedef()||retType.IsReference()||retType.IsPointer()) retType = retType.ToType();
    //    retType = Type::ByName(retType.Name());
    cerr << ">>> now type is " << retType.Name() << ", is reference: " << 
      retType.IsReference() << endl;
   ret = Object(retType, addr);
  }
  if(retType.IsPointer()) {
    cerr << ">>> return type is a pointer to " << retType.Name() << endl; 
    while(retType.IsTypedef()||retType.IsReference()||retType.IsPointer()) retType = retType.ToType();
    //    retType = Type::ByName(retType.Name());
    cerr << ">>> now type is " << retType.Name() << ", is pointer: " << 
      retType.IsPointer() << endl;
    ret = Object(retType, addr);
  }
  cerr << ">>> invoked method returned object of type " << ret.TypeOf().Name() << endl;
  if(!ret) 
     throw edm::Exception(edm::errors::Configuration)
      << "method \"" << method_.Name() 
      << "\" returned void invoked on object of type " 
      << o.TypeOf().Name() << "\n";
  return ret;
}
