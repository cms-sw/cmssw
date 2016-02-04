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
//

// system include files

// user include files
#include "FWCore/Framework/interface/GenericHandle.h"

namespace edm {
void convert_handle(BasicHandle const& orig,
                    Handle<GenericObject>& result)
{
  if(orig.failedToGet()) {
    result.setWhyFailed(orig.whyFailed());
    return;
  }
  WrapperHolder originalWrap = orig.wrapperHolder();
  if(!originalWrap.isValid()) {
    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << "edm::BasicHandle has null pointer to Wrapper";
  }
  
  Reflex::Object wrap(Reflex::Type::ByTypeInfo(originalWrap.wrappedTypeInfo()), const_cast<void*>(originalWrap.wrapper()));
  assert(wrap != Reflex::Object());
  
  Reflex::Object product(wrap.Get("obj"));
  if(!product){
    throw edm::Exception(edm::errors::LogicError)<<"GenericObject could not find 'obj' member";
  }
  if(product.TypeOf().IsTypedef()){
    //For a 'Reflex::Typedef' the 'ToType' method returns the actual type
    // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
    // only for a 'real' class
    product = Reflex::Object(product.TypeOf().ToType(), product.Address());
    assert(!product.TypeOf().IsTypedef());
  }
  //NOTE: comparing on type doesn't seem to always work! The problem appears to be if we have a typedef
  if(product.TypeOf()!=result.type() &&
     !product.TypeOf().IsEquivalentTo(result.type()) &&
     product.TypeOf().TypeInfo()!= result.type().TypeInfo()){
    throw edm::Exception(edm::errors::LogicError)<<"GenericObject asked for "<<result.type().Name()
    <<" but was given a "<<product.TypeOf().Name();
  }
  
  Handle<GenericObject> h(product, orig.provenance(), orig.id());
  h.swap(result);
}

///Specialize the getByLabel method to work with a Handle<GenericObject>
template<>
bool
edm::Event::getByLabel<GenericObject>(std::string const& label,
                                      std::string const& productInstanceName,
                                      Handle<GenericObject>& result) const
{
  BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.type().TypeInfo()), label, productInstanceName, std::string());
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
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.type().TypeInfo()), tag.label(), tag.instance(),tag.process());
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      addToGotBranchIDs(*bh.provenance());
      return true;
    }
  }
  return false;
}

}
