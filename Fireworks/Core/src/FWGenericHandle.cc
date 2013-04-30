// -*- C++ -*-
//
// Package:     Framework
// Class  :     FWGenericHandle
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Mar 30 15:48:37 EST 2006
//

// system include files

// user include files
#include "Fireworks/Core/src/FWGenericHandle.h"

namespace edm {
void convert_handle(BasicHandle const& orig,
                    Handle<FWGenericObject>& result)
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
    throw edm::Exception(edm::errors::LogicError)<<"FWGenericObject could not find 'obj' member";
  }
  if(product.TypeOf().IsTypedef()){
    //For a 'Reflex::Typedef' the 'ToType' method returns the actual type
    // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
    // only for a 'real' class
    product = Reflex::Object(product.TypeOf().ToType(), product.Address());
    assert(!product.TypeOf().IsTypedef());
  }
  // NOTE: comparing on type doesn't seem to always work! The problem appears to be if we have a typedef
  if(product.TypeOf()!=result.type() &&
     !product.TypeOf().IsEquivalentTo(result.type()) &&
     product.TypeOf().TypeInfo()!= result.type().TypeInfo()){
        std::cerr << "FWGenericObject asked for "<<result.type().Name()
         <<" but was given a " << product.TypeOf().Name();
    throw edm::Exception(edm::errors::LogicError)<<"FWGenericObject asked for "<<result.type().Name()
    <<" but was given a "<<product.TypeOf().Name();
  }
  
  Handle<FWGenericObject> h(product, orig.provenance(), orig.id());
  h.swap(result);
}

///Specialize the getByLabel method to work with a Handle<FWGenericObject>
template<>
bool
edm::EventBase::getByLabel<FWGenericObject>(edm::InputTag const& tag,
                                             Handle<FWGenericObject>& result) const
{
   std::string dataTypeName = result.type().Name(Reflex::SCOPED);
   if (dataTypeName[dataTypeName.size() -1] == '>')
      dataTypeName += " ";
   std::string wrapperName = "edm::Wrapper<" + dataTypeName + ">";

   Reflex::Type wrapperType = Reflex::Type::ByName(wrapperName);

   BasicHandle bh = this->getByLabelImpl(wrapperType.TypeInfo(),
                                         result.type().TypeInfo(),
                                         tag);
   convert_handle(bh, result);  // throws on conversion error
   if(bh.failedToGet()) 
      return false;
   return true;
}

}
