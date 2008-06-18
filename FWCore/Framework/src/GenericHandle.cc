// -*- C++ -*-
//
// Package:     Framework
// Class  :     GenericHandle
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Mar 30 15:48:37 EST 2006
// $Id: GenericHandle.cc,v 1.8 2008/05/14 19:45:13 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/GenericHandle.h"

namespace edm {
void convert_handle(BasicHandle const& orig,
                    Handle<GenericObject>& result)
{
  using namespace ROOT::Reflex;
  if(orig.failedToGet()) {
    result.setWhyFailed(orig.whyFailed());
    return;
  }
  EDProduct const* originalWrap = orig.wrapper();
  if (originalWrap == 0)
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << "edm::BasicHandle has null pointer to Wrapper";
  
  //Since a pointer to an EDProduct is not necessarily the same as a pointer to the actual type
  // (compilers are allowed to offset the two) we must get our object via a two step process
  Object edproductObject(Type::ByTypeInfo(typeid(EDProduct)), const_cast<EDProduct*>(originalWrap));
  assert(edproductObject != Object());
  
  Object wrap(edproductObject.CastObject(edproductObject.DynamicType()));
  assert(wrap != Object());
  
  Object product(wrap.Get("obj"));
  if(!product){
    throw edm::Exception(edm::errors::LogicError)<<"GenericObject could not find 'obj' member";
  }
  if(product.TypeOf().IsTypedef()){
    //For a 'Reflex::Typedef' the 'ToType' method returns the actual type
    // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
    // only for a 'real' class
    product = Object(product.TypeOf().ToType(), product.Address());
    assert(!product.TypeOf().IsTypedef());
  }
  //NOTE: comparing on type doesn't seem to always work! The problem appears to be if we have a typedef
  if(product.TypeOf()!=result.type() &&
     !product.TypeOf().IsEquivalentTo(result.type()) &&
     product.TypeOf().TypeInfo()!= result.type().TypeInfo()){
    throw edm::Exception(edm::errors::LogicError)<<"GenericObject asked for "<<result.type().Name()
    <<" but was given a "<<product.TypeOf().Name();
  }
  
  Handle<GenericObject> h(product, orig.provenance());
  h.swap(result);
}

///Specialize the getByLabel method to work with a Handle<GenericObject>
template<>
bool
edm::Event::getByLabel<GenericObject>(std::string const& label,
                                      const std::string& productInstanceName,
                                      Handle<GenericObject>& result) const
{
  BasicHandle bh = this->getByLabel_(TypeID(result.type().TypeInfo()), label, productInstanceName);
  convert_handle(bh, result);  // throws on conversion error
  if(!bh.failedToGet()) {
    addToGotBranchIDs(*bh.provenance());
    return true;
  }
  return false;
}

template<>
bool
edm::Event::getByLabel<GenericObject>(edm::InputTag const& tag,
                                             Handle<GenericObject>& result) const
{
  if (tag.process().empty()) {
    return this->getByLabel(tag.label(), tag.instance(), result);
  } else {
    BasicHandle bh = this->getByLabel_(TypeID(result.type().TypeInfo()), tag.label(), tag.instance(),tag.process());
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      addToGotBranchIDs(*bh.provenance());
      return true;
    }
  }
  return false;
}

}
