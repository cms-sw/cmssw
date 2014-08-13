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
void convert_handle(BasicHandle && orig,
                    Handle<GenericObject>& result)
{
  if(orig.failedToGet()) {
    result.setWhyFailedFactory(orig.whyFailedFactory());
    return;
  }
  EDProduct const* originalWrap = orig.wrapper();
  if(originalWrap == nullptr) {
    throw Exception(errors::InvalidReference,"NullPointer")
      << "edm::BasicHandle has null pointer to Wrapper";
  }
  
  ObjectWithDict wrap(originalWrap->wrappedTypeInfo(), const_cast<EDProduct*>(originalWrap));
  assert(bool(wrap));
  
  ObjectWithDict product(wrap.get("obj"));
  if(!product){
    throw Exception(errors::LogicError)<<"GenericObject could not find 'obj' member";
  }
  if(product.typeOf() != result.type()) {
    throw Exception(errors::LogicError) << "GenericObject asked for " << result.type().name()
      << " but was given a " << product.typeOf().name();
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
  BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.type().typeInfo()), label, productInstanceName, std::string(), moduleCallingContext_);
  convert_handle(std::move(bh), result);  // throws on conversion error
  if(!result.failedToGet()) {
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
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.type().typeInfo()), tag.label(), tag.instance(),tag.process(), moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if(!result.failedToGet()) {
      addToGotBranchIDs(*bh.provenance());
      return true;
    }
  }
  return false;
}

}
