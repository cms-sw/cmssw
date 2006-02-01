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
    The GenericHandle allows one to get data back from the edm::Event as a ROOT::Reflex::Object instead
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
// $Id: GenericHandle.h,v 1.1 2006/01/10 17:21:35 chrjones Exp $
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
        type_(ROOT::Reflex::Type::ByName(iName)),prov_(0),id_(0) {
           if(type_ == ROOT::Reflex::Type()) {
              throw edm::Exception(edm::errors::NotFound)<<"Handle<GenericObject> told to use uknown type '"<<iName<<"'.\n Please check spelling or that a module uses this type in the job.";
           }
           if(type_.IsTypedef()){
              //For a 'Reflex::Typedef' the 'toType' method returns the actual type
              // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
              // only for a 'real' class
              type_ = type_.ToType();
           }
        }
   
   ///Throws exception if iType is invalid
   Handle(const ROOT::Reflex::Type& iType):
      type_(iType),prov_(0),id_(0) {
         if(iType == ROOT::Reflex::Type()) {
            throw edm::Exception(edm::errors::NotFound)<<"Handle<GenericObject> given an invalid ROOT::Reflex::Type";
         }
         if(type_.IsTypedef()){
            //For a 'Reflex::Typedef' the 'toType' method returns the actual type
            // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
            // only for a 'real' class
            type_ = type_.ToType();
         }
      }
   
   Handle(const Handle<GenericObject>& h):
   type_(h.type_),
   prod_(h.prod_),
   prov_(h.prov_),
   id_(h.id_)
   { }
   
   Handle(ROOT::Reflex::Object const& prod, Provenance const* prov):
   type_(prod.TypeOf()),
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
      
   ROOT::Reflex::Object const* product() const {return &prod_;}
   ROOT::Reflex::Object const* operator->() const {return &prod_;}
   ROOT::Reflex::Object const& operator*() const {return prod_;}
   
   ROOT::Reflex::Type const& type() const {return type_;}
   Provenance const* provenance() const {return prov_;}
   
   ProductID id() const {return id_;}
      
private:
   ROOT::Reflex::Type type_;
   ROOT::Reflex::Object prod_;
   Provenance const* prov_;    
   ProductID id_;
};

typedef Handle<GenericObject> GenericHandle;

///specialize this function for GenericHandle
void convert_handle(BasicHandle const& orig,
                    Handle<GenericObject>& result)
{
   using namespace ROOT::Reflex;
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

///Specialize the Event's getByLabel method to work with a Handle<GenericObject>
template<>
void
edm::Event::getByLabel<GenericObject>(std::string const& label,
                                      const std::string& productInstanceName,
                                      Handle<GenericObject>& result) const
{
   BasicHandle bh = this->getByLabel_(TypeID(result.type().TypeInfo()), label, productInstanceName);
   gotProductIDs_.push_back(bh.id());
   convert_handle(bh, result);  // throws on conversion error
}


}
#endif
