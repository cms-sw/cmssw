#ifndef Framework_ConstProductRegistry_h
#define Framework_ConstProductRegistry_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ConstProductRegistry
// 
/**\class ConstProductRegistry ConstProductRegistry.h FWCore/Framework/interface/ConstProductRegistry.h

 Description: Provides a 'service' interface to the ProductRegistry

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 22 18:01:21 CEST 2005
// $Id: ConstProductRegistry.h,v 1.1 2005/09/22 16:19:34 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/connect_but_block_self.h"

// forward declarations
namespace edm {
   class ConstProductRegistry
  {

   public:
     typedef ProductRegistry::ProductList ProductList;
     
     ConstProductRegistry(SignallingProductRegistry& iReg) : reg_(&iReg) {}
     //virtual ~ConstProductRegistry();
     
      // ---------- const member functions ---------------------
     ProductList const& productList() const {return reg_->productList();}
     
     unsigned long nextID() const {return reg_->nextID();}
     
     template< class T>
        void watchProductAdditions(const T& iFunc){
           serviceregistry::connect_but_block_self( reg_->productAddedSignal_, iFunc);
        }
     template< class T, class TMethod>
        void watchProductAdditions(T& iObj, TMethod iMethod){
           serviceregistry::connect_but_block_self(reg_->productAddedSignal_, boost::bind(iMethod, iObj,_1));
        }
     
   private:
      ConstProductRegistry(const ConstProductRegistry&); // stop default

      const ConstProductRegistry& operator=(const ConstProductRegistry&); // stop default

      // ---------- member data --------------------------------
      SignallingProductRegistry* reg_;
  };
}

#endif
