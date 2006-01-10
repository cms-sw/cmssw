#ifndef Framework_GenericHandle_h
#define Framework_GenericHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     GenericHandle
// 
/**\class GenericHandle GenericHandle.h FWCore/Framework/interface/GenericHandle.h

 Description: Allows interaction with data in the Event without actually using the C++ class

 Usage:
    The GenericHandle allows one to get data back from the edm::Event as a seal::reflex::Object instead
  of as the actual C++ class type.

  //make a handle to hold an instance of 'MyClass'
  edm::GenericHandle myHandle("MyClass");

  event.getByLabel("mine",myHandle);

  //call the print method of 'MyClass' instance
  myHandle->invoke("print);  
*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jan  7 15:40:43 EST 2006
// $Id$
//

// system include files
#include <string>

// user include files
#include "Reflex/Object.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

// forward declarations
namespace edm {
   ///This class is just a 'tag' used to allow a specialization of edm::Handle
struct GenericObject
{
};

template<>
class Handle<GenericObject> {
public:
      ///Throws exception if iName is not a known C++ class type
      Handle(const std::string& iName) : 
        type_(seal::reflex::Type::byName(iName)),prov_(0),id_(0) {
           if(type_ == seal::reflex::Type()) {
              throw edm::Exception(edm::errors::NotFound)<<"Handle<GenericObject> told to use uknown type '"<<iName<<"'.\n Please check spelling or that a module uses this type in the job.";
           }
           if(type_.isTypedef()){
              //For a 'reflex::Typedef' the 'toType' method returns the actual type
              // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
              // only for a 'real' class
              type_ = type_.toType();
           }
        }
   
   ///Throws exception if iType is invalid
   Handle(const seal::reflex::Type& iType):
      type_(iType),prov_(0),id_(0) {
         if(iType == seal::reflex::Type()) {
            throw edm::Exception(edm::errors::NotFound)<<"Handle<GenericObject> given an invalid seal::reflex::Type";
         }
         if(type_.isTypedef()){
            //For a 'reflex::Typedef' the 'toType' method returns the actual type
            // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
            // only for a 'real' class
            type_ = type_.toType();
         }
      }
   
   Handle(const Handle<GenericObject>& h):
   type_(h.type_),
   prod_(h.prod_),
   prov_(h.prov_),
   id_(h.id_)
   { }
   
   Handle(seal::reflex::Object const& prod, Provenance const* prov):
   type_(prod.type()),
   prod_(prod),
   prov_(prov),
   id_(prov->event.productID_) { 
      assert(prod_);
      assert(prov_);
      assert(id_ != ProductID());
   }
   
      
      //~Handle();
      
   void swap(Handle<GenericObject>& other)
   {
      std::swap(type_, other.type_);
      std::swap(prod_, other.prod_);
      std::swap(prov_, other.prov_);
      std::swap(id_, other.id_);
   }
   
   
   Handle<GenericObject>& operator=(const Handle<GenericObject>& rhs)
   {
      Handle<GenericObject> temp(rhs);
      this->swap(temp);
      return *this;
   }
   
   bool isValid() const {
      return prod_ && 0!= prov_;
   }
      
   seal::reflex::Object const* product() const {return &prod_;}
   seal::reflex::Object const* operator->() const {return &prod_;}
   seal::reflex::Object const& operator*() const {return prod_;}
   
   seal::reflex::Type const& type() const {return type_;}
   Provenance const* provenance() const {return prov_;}
   
   ProductID id() const {return id_;}
      
private:
   seal::reflex::Type type_;
   seal::reflex::Object prod_;
   Provenance const* prov_;    
   ProductID id_;
};

typedef Handle<GenericObject> GenericHandle;

///specialize this function for GenericHandle
void convert_handle(BasicHandle const& orig,
                    Handle<GenericObject>& result)
{
   using namespace seal::reflex;
   EDProduct const* originalWrap = orig.wrapper();
   if (originalWrap == 0)
      throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
         << "edm::BasicHandle has null pointer to Wrapper";

   //Since a pointer to an EDProduct is not necessarily the same as a pointer to the actual type
   // (compilers are allowed to offset the two) we must get our object via a two step process
   Object edproductObject(Type::byTypeInfo(typeid(EDProduct)), const_cast<EDProduct*>(originalWrap));
   assert(edproductObject != Object());

   Object wrap(edproductObject.castObject(edproductObject.dynamicType()));
   assert(wrap != Object());
   
   Object product(wrap.get("obj"));
   if(!product){
      throw edm::Exception(edm::errors::LogicError)<<"GenericObject could not find 'obj' member";
   }
   if(product.type().isTypedef()){
      //For a 'reflex::Typedef' the 'toType' method returns the actual type
      // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
      // only for a 'real' class
      product = Object(product.type().toType(), product.address());
      assert(!product.type().isTypedef());
   }
   //NOTE: comparing on type doesn't seem to always work! The problem appears to be if we have a typedef
   if(product.type()!=result.type() &&
      !product.type().isEquivalentTo(result.type()) &&
      product.type().typeInfo()!= result.type().typeInfo()){
      throw edm::Exception(edm::errors::LogicError)<<"GenericObject asked for "<<result.type().name()
      <<" but was given a "<<product.type().name();
   }
   
   Handle<GenericObject> h(product, orig.provenance());
   h.swap(result);
}

///Specialize the Event's getByLabel method to work with a Handle<GenericObject>
template<>
void
edm::Event::getByLabel<GenericObject>(std::string const& label,
                                      const std::string& productInstanceName,
                                      Handle<GenericObject>& result) const
{
   BasicHandle bh = this->getByLabel_(TypeID(result.type().typeInfo()), label, productInstanceName);
   gotProductIDs_.push_back(bh.id());
   convert_handle(bh, result);  // throws on conversion error
}


}
#endif
