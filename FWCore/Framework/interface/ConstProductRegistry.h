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
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProductRegistry.h"

// forward declarations
namespace edm {
   class ConstProductRegistry
  {

   public:
     typedef ProductRegistry::ProductList ProductList;
     
     ConstProductRegistry(const ProductRegistry& iReg) : reg_(&iReg) {}
     //virtual ~ConstProductRegistry();
     
      // ---------- const member functions ---------------------
     ProductList const& productList() const {return reg_->productList();}
     
     unsigned long nextID() const {return reg_->nextID();}
     
   private:
      ConstProductRegistry(const ConstProductRegistry&); // stop default

      const ConstProductRegistry& operator=(const ConstProductRegistry&); // stop default

      // ---------- member data --------------------------------
      const ProductRegistry* reg_;
  };
}

#endif
