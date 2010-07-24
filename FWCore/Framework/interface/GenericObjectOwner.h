#ifndef FWCore_Framework_GenericObjectOwner_h
#define FWCore_Framework_GenericObjectOwner_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     GenericObjectOwner
// 
/**\class GenericObjectOwner GenericObjectOwner.h FWCore/Framework/interface/GenericObjectOwner.h

 Description: Helper classed used for doing a 'generic' put into the edm::Event

 Usage:
    

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Feb  3 19:43:16 EST 2008
//

// system include files
#include <string>
#include "Reflex/Object.h"

// user include files
#include "FWCore/Framework/interface/Event.h"

// forward declarations
namespace edm {
class GenericObjectOwner
{

   public:
      GenericObjectOwner(): m_ownData(false){}
      explicit GenericObjectOwner(Reflex::Object const& iObject,
                                  bool iOwnData=true):
         m_object(iObject), m_ownData(iOwnData) {}
      ~GenericObjectOwner();

      // ---------- const member functions ---------------------
      Reflex::Object object() const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void swap(GenericObjectOwner&);
      void release();
   
   private:
      GenericObjectOwner(GenericObjectOwner const&); // stop default

      GenericObjectOwner const& operator=(GenericObjectOwner const&); // stop default

      // ---------- member data --------------------------------
      Reflex::Object m_object;
      bool m_ownData;
};

   //Need to specialize OrphanHandle because we don't actually have a long lived GenericObjectOwner
   template <>
   class OrphanHandle<GenericObjectOwner> {
   public:
      // Default constructed handles are invalid.
      OrphanHandle() {}
      
      OrphanHandle(OrphanHandle<GenericObjectOwner> const& h):
      prod_(h.prod_.object(),false), id_(h.id_) {}
      
      OrphanHandle(Reflex::Object const& prod, ProductID const& id):
      prod_(prod,false), id_(id) {}
      
      //~OrphanHandle();
      
      void swap(OrphanHandle<GenericObjectOwner>& other){
         prod_.swap(other.prod_);
         std::swap(id_,other.id_);
      }
      
      
      OrphanHandle<GenericObjectOwner>& operator=(OrphanHandle<GenericObjectOwner> const& rhs)
      {
         OrphanHandle<GenericObjectOwner> temp(rhs);
         swap(temp);
	 return *this;
      }
      
      bool isValid() const {return 0 !=prod_.object().Address();}
         
      GenericObjectOwner const* product() const {return &prod_;}
      GenericObjectOwner const* operator->() const {return product();}
      GenericObjectOwner const& operator*() const {return prod_;}
      
      ProductID id() const {return id_;}
         
         
      private:
         GenericObjectOwner prod_;
         ProductID id_;
      };
      
   template<>
   OrphanHandle<GenericObjectOwner> 
   Event::put<GenericObjectOwner>(std::auto_ptr<GenericObjectOwner> product, std::string const& productInstanceName);
   
}
#endif
