#ifndef FWCore_Framework_GenericHandle_h
#define FWCore_Framework_GenericHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     GenericHandle
// 
/**\class GenericHandle GenericHandle.h FWCore/Framework/interface/GenericHandle.h

 Description: Allows interaction with data in the Event without actually using the C++ class

 Usage:
    The GenericHandle allows one to get data back from the edm::Event as an ObjectWithDict instead
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
//

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

// system include files
#include <string>

// forward declarations
namespace edm {
   ///This class is just a 'tag' used to allow a specialization of edm::Handle
struct GenericObject {
};

template<>
class Handle<GenericObject> {
public:
    ///Throws exception if iName is not a known C++ class type
    Handle(std::string const& iName) : 
       type_(TypeWithDict::byName(iName)), prod_(), prov_(nullptr) {
          if(!bool(type_)) {
             Exception::throwThis(errors::NotFound,
             "Handle<GenericObject> told to use uknown type '",
             iName.c_str(),
             "'.\n Please check spelling or that a module uses this type in the job.");
           }
           if(type_.isTypedef()){
              //For a 'Typedef' the 'toType' method returns the actual type
              // this is needed since you are not allowed to 'invoke' methods of a 'Typedef'
              // only for a 'real' class
              type_ = type_.toType();
           }
        }
   
   ///Throws exception if iType is invalid
   Handle(TypeWithDict const& iType) :
      type_(iType), prod_(), prov_(nullptr) {
         if(!bool(iType)) {
            Exception::throwThis(errors::NotFound, "Handle<GenericObject> given an invalid type");
         }
         if(type_.isTypedef()){
            //For a 'Typedef' the 'toType' method returns the actual type
            // this is needed since you are now allowed to 'invoke' methods of a 'Typedef'
            // only for a 'real' class
            type_ = type_.toType();
         }
      }
   
   Handle(Handle<GenericObject> const& h):
   type_(h.type_),
   prod_(h.prod_),
   prov_(h.prov_),
   whyFailed_(h.whyFailed_) {
   }
   
   Handle(ObjectWithDict const& prod, Provenance const* prov, ProductID const&):
   type_(prod.typeOf()),
   prod_(prod),
   prov_(prov) { 
      assert(prod_);
      assert(prov_);
   }
   
      //~Handle();
      
   void swap(Handle<GenericObject>& other) {
      // use unqualified swap for user defined classes
      using std::swap;
      swap(type_, other.type_);
      std::swap(prod_, other.prod_);
      swap(prov_, other.prov_);
      swap(whyFailed_, other.whyFailed_);
   }
   
   
   Handle<GenericObject>& operator=(Handle<GenericObject> const& rhs) {
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
   ObjectWithDict const* product() const { 
     if(this->failedToGet()) { 
       whyFailed_->raise();
     } 
     return &prod_;
   }
   ObjectWithDict const* operator->() const {return this->product();}
   ObjectWithDict const& operator*() const {return *(this->product());}
   
   TypeWithDict const& type() const {return type_;}
   Provenance const* provenance() const {return prov_;}
   
   ProductID id() const {return prov_->productID();}

   void clear() { prov_ = 0; whyFailed_.reset();}
      
   void setWhyFailed(boost::shared_ptr<cms::Exception> const& iWhyFailed) {
    whyFailed_=iWhyFailed;
  }
private:
   TypeWithDict type_;
   ObjectWithDict prod_;
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
