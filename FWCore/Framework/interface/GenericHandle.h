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
// $Id: GenericHandle.h,v 1.5 2006/08/31 23:26:24 wmtan Exp $
//

// system include files
#include <string>

// user include files
#include "Reflex/Object.h"
#include "FWCore/Framework/interface/DataViewImpl.h"
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
        type_(ROOT::Reflex::Type::ByName(iName)), prod_(), prov_(0), id_(0) {
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
      type_(iType), prod_(), prov_(0), id_(0) {
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
      // use unqualified swap for user defined classes
      using std::swap;
      swap(type_, other.type_);
      std::swap(prod_, other.prod_);
      swap(prov_, other.prov_);
      swap(id_, other.id_);
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
                    Handle<GenericObject>& result);


///Specialize the Event's getByLabel method to work with a Handle<GenericObject>
template<>
void
edm::DataViewImpl::getByLabel<GenericObject>(std::string const& label,
                                      const std::string& productInstanceName,
                                      Handle<GenericObject>& result) const;

}
#endif
