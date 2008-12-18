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
    The GenericHandle allows one to get data back from the edm::Event as a Reflex::Object instead
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
// $Id: GenericHandle.h,v 1.13.4.5 2008/12/16 00:43:04 wmtan Exp $
//

// system include files
#include <string>

// user include files
#include "Reflex/Object.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/UseReflex.h"
#include "DataFormats/Common/interface/Handle.h"

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
      Handle(std::string const& iName) : 
        type_(Reflex::Type::ByName(iName)), prod_(), prov_(0) {
           if(type_ == Reflex::Type()) {
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
   Handle(Reflex::Type const& iType):
      type_(iType), prod_(), prov_(0) {
         if(iType == Reflex::Type()) {
            throw edm::Exception(edm::errors::NotFound)<<"Handle<GenericObject> given an invalid Reflex::Type";
         }
         if(type_.IsTypedef()){
            //For a 'Reflex::Typedef' the 'toType' method returns the actual type
            // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
            // only for a 'real' class
            type_ = type_.ToType();
         }
      }
   
   Handle(Handle<GenericObject> const& h):
   type_(h.type_),
   prod_(h.prod_),
   prov_(h.prov_),
   whyFailed_(h.whyFailed_)
   { }
   
   Handle(Reflex::Object const& prod, Provenance const* prov, ProductID const& pid):
   type_(prod.TypeOf()),
   prod_(prod),
   prov_(prov) { 
      assert(prod_);
      assert(prov_);
      assert(prov_->productID() != ProductID());
   }
   
      //~Handle();
      
   void swap(Handle<GenericObject>& other)
   {
      // use unqualified swap for user defined classes
      using std::swap;
      swap(type_, other.type_);
      std::swap(prod_, other.prod_);
      swap(prov_, other.prov_);
      swap(whyFailed_, other.whyFailed_);
   }
   
   
   Handle<GenericObject>& operator=(Handle<GenericObject> const& rhs)
   {
      Handle<GenericObject> temp(rhs);
      this->swap(temp);
      return *this;
   }
   
   bool isValid() const {
      return prod_ && 0!= prov_;
   }

   bool failedToGet() const {
     return 0 != whyFailed_.get();
   }
   Reflex::Object const* product() const { 
     if(this->failedToGet()) { 
       throw *whyFailed_;
     } 
     return &prod_;
   }
   Reflex::Object const* operator->() const {return this->product();}
   Reflex::Object const& operator*() const {return *(this->product());}
   
   Reflex::Type const& type() const {return type_;}
   Provenance const* provenance() const {return prov_;}
   
   ProductID id() const {return prov_->productID();}

   void clear() { prov_ = 0; whyFailed_.reset();}
      
   void setWhyFailed(boost::shared_ptr<cms::Exception> const& iWhyFailed) {
    whyFailed_=iWhyFailed;
  }
private:
   Reflex::Type type_;
   Reflex::Object prod_;
   Provenance const* prov_;    
   boost::shared_ptr<cms::Exception> whyFailed_;

};

typedef Handle<GenericObject> GenericHandle;

///specialize this function for GenericHandle
void convert_handle(BasicHandle const& orig,
                    Handle<GenericObject>& result);


///Specialize the Event's getByLabel method to work with a Handle<GenericObject>
template<>
bool
edm::Event::getByLabel<GenericObject>(std::string const& label,
                                      std::string const& productInstanceName,
                                      Handle<GenericObject>& result) const;

template <> 	 
bool 	 
edm::Event::getByLabel(edm::InputTag const& tag, Handle<GenericObject>& result) const; 	 

}
#endif
