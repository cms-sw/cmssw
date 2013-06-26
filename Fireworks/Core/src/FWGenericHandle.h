#ifndef Fireworks_FWGenericHandle_h
#define Fireworks_FWGenericHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     FWGenericHandle
// 
/**\class FWGenericHandle FWGenericHandle.h Fireworks/Core/interface/FWGenericHandle.h

 Description: Allows interaction with data in the Event without actually using 
              the C++ class. Ported to work with 

 Usage:
    This is a rip-off of edm::FWGenericHandle. I extended it to work with 
    edm::EventBase as well.

    The FWGenericHandle allows one to get data back from the edm::EventBase as 
    a edm::ObjectWithDict instead of as the actual C++ class type.

    //make a handle to hold an instance of 'MyClass'
    edm::FWGenericHandle myHandle("MyClass");
    
    event.getByLabel("mine",myHandle);
    
    //call the print method of 'MyClass' instance
    myHandle->invoke("print);  
*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jan  7 15:40:43 EST 2006
// $Id: FWGenericHandle.h,v 1.4 2013/02/10 22:12:04 wmtan Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"

// forward declarations
namespace edm {
   ///This class is just a 'tag' used to allow a specialization of edm::Handle
struct FWGenericObject
{
};

template<>
class Handle<FWGenericObject> {
public:
      ///Throws exception if iName is not a known C++ class type
      Handle(std::string const& iName) : 
        type_(edm::TypeWithDict::byName(iName)), prod_(), prov_(0) {
           if(type_ == edm::TypeWithDict()) {
              Exception::throwThis(errors::NotFound,
                "Handle<FWGenericObject> told to use uknown type '",
                iName.c_str(),
                "'.\n Please check spelling or that a module uses this type in the job.");
           }
        }
   
   ///Throws exception if iType is invalid
   Handle(edm::TypeWithDict const& iType):
      type_(iType), prod_(), prov_(0) {
         if(iType == edm::TypeWithDict()) {
            Exception::throwThis(errors::NotFound, "Handle<FWGenericObject> given an invalid edm::TypeWithDict");
         }
      }
   
   Handle(Handle<FWGenericObject> const& h):
   type_(h.type_),
   prod_(h.prod_),
   prov_(h.prov_),
   whyFailed_(h.whyFailed_)
   { }
   
   Handle(edm::ObjectWithDict const& prod, Provenance const* prov, ProductID const& pid):
   type_(prod.typeOf()),
   prod_(prod),
   prov_(prov) { 
      assert(prod_);
      assert(prov_);
      // assert(prov_->productID() != ProductID());
   }
   
      //~Handle();
      
   void swap(Handle<FWGenericObject>& other)
   {
      // use unqualified swap for user defined classes
      using std::swap;
      swap(type_, other.type_);
      std::swap(prod_, other.prod_);
      swap(prov_, other.prov_);
      swap(whyFailed_, other.whyFailed_);
   }
   
   
   Handle<FWGenericObject>& operator=(Handle<FWGenericObject> const& rhs)
   {
      Handle<FWGenericObject> temp(rhs);
      this->swap(temp);
      return *this;
   }
   
   bool isValid() const {
      return prod_ && 0!= prov_;
   }

   bool failedToGet() const {
     return 0 != whyFailed_.get();
   }
   edm::ObjectWithDict const* product() const { 
     if(this->failedToGet()) { 
       whyFailed_->raise();
     } 
     return &prod_;
   }
   edm::ObjectWithDict const* operator->() const {return this->product();}
   edm::ObjectWithDict const& operator*() const {return *(this->product());}
   
   edm::TypeWithDict const& type() const {return type_;}
   Provenance const* provenance() const {return prov_;}
   
   ProductID id() const {return prov_->productID();}

   void clear() { prov_ = 0; whyFailed_.reset();}
      
   void setWhyFailed(boost::shared_ptr<cms::Exception> const& iWhyFailed) {
    whyFailed_=iWhyFailed;
  }
private:
   edm::TypeWithDict type_;
   edm::ObjectWithDict prod_;
   Provenance const* prov_;    
   boost::shared_ptr<cms::Exception> whyFailed_;
};

typedef Handle<FWGenericObject> FWGenericHandle;

///specialize this function forFWGenericHandle
void convert_handle(BasicHandle const& orig,
                    Handle<FWGenericObject>& result);


///Specialize the Event's getByLabel method to work with a Handle<FWGenericObject>
template <>
bool
edm::EventBase::getByLabel(edm::InputTag const& tag, Handle<FWGenericObject>& result) const;   

}
#endif
