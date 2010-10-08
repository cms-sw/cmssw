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
// $Id: ConstProductRegistry.h,v 1.7 2008/12/18 04:49:01 wmtan Exp $
//

// system include files
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/connect_but_block_self.h"

// forward declarations
namespace edm {
  class ConstProductRegistry
  {

  public:
    typedef ProductRegistry::ProductList ProductList;
     
    ConstProductRegistry(SignallingProductRegistry& iReg) : reg_(&iReg) { }
     
    // ---------- const member functions ---------------------
    ProductRegistry const& productRegistry() const {return *reg_;}

    ProductList const& productList() const {return reg_->productList();}

    // Return all the branch names currently known to *this.  This
    // does a return-by-value of the vector so that it may be used in
    // a colon-initialization list.
    std::vector<std::string> allBranchNames() const {return reg_->allBranchNames();}

    // Return pointers to (const) BranchDescriptions for all the
    // BranchDescriptions known to *this.  This does a
    // return-by-value of the vector so that it may be used in a
    // colon-initialization list.
    std::vector<BranchDescription const*> allBranchDescriptions() const {return reg_->allBranchDescriptions();}

    bool anyProductProduced() const {return reg_->anyProductProduced();}
     
    template< class T>
    void watchProductAdditions(const T& iFunc)
    {
      serviceregistry::connect_but_block_self(reg_->productAddedSignal_, 
					      iFunc);
    }
    template< class T, class TMethod>
    void watchProductAdditions(T& iObj, TMethod iMethod)
    {
      serviceregistry::connect_but_block_self(reg_->productAddedSignal_, 
					      boost::bind(iMethod, iObj,_1));
    }
     
  private:
    // stop default
    ConstProductRegistry(const ConstProductRegistry&); 

    // stop default
    const ConstProductRegistry& operator=(const ConstProductRegistry&); 

    // ---------- member data --------------------------------
    SignallingProductRegistry* reg_;
  };
}

#endif
