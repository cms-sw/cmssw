/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.6 2006/08/01 05:34:43 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.6 2006/08/01 05:34:43 wmtan Exp $";


#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

namespace edm {
  void
  ProductRegistry::addProduct(BranchDescription const& productDesc, bool fromListener) {
    throwIfFrozen();
    productDesc.init();
    productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    addCalled(productDesc,fromListener);
  }
  
  void
  ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    throwIfFrozen();
    productDesc.init();
    productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    if (productDesc.productID().id_ >= nextID_) {
      nextID_ = productDesc.productID().id_ + 1;
    }
  }
  
  void
  ProductRegistry::setProductIDs() {
    throwIfFrozen();
    for (ProductList::iterator it = productList_.begin(); it != productList_.end(); ++it) {
       if (it->second.productID().id_ == 0) {
          it->second.productID_.id_ = nextID_++;
       }
    }
    frozen_ = true;
  }
  
  void
  ProductRegistry::setFrozen() const {
    if (frozen_) return;
/*
    for (ProductList::const_iterator it = productList_.begin(); it != productList_.end(); ++it) {
      if (it->second.productID_.id_ == 0) {
       throw cms::Exception("ProductRegistry", "setFrozen")
          << "cannot read the ProductRegistry because it is not yet frozen.";
      }
    }
*/
    frozen_ = true;
  }
  
  void
  ProductRegistry::throwIfFrozen() const {
    if (frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
            << "cannot modify the ProductRegistry because it is frozen";
    }
  }
  
  void
  ProductRegistry::throwIfNotFrozen() const {
/*
    if (!frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
            << "cannot read the ProductRegistry because it is not yet frozen";
    }
*/
  }
  
  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
  }

  bool
  ProductRegistry::merge(ProductRegistry const& other, BranchDescription::MatchMode m) {
    if (nextID() < other.nextID()) return false;
    if (size() < other.size()) return false;
    ProductRegistry::ProductList::const_iterator j = productList().begin();
    ProductRegistry::ProductList::const_iterator s = productList().end();
    int nProduced = 0;
    for ( ; j != s; ++j) {
      if(j->second.produced()) ++nProduced;
    }
    if (size() != (other.size() + nProduced)) return false;
    ProductRegistry::ProductList::const_iterator i = other.productList().begin();
    ProductRegistry::ProductList::const_iterator e = other.productList().end();

    j = productList().begin();
    for( ; i != e; ++i, ++j) {
      while (j->second.produced()) ++j;
      if (i->first != j->first) return false;
    }
    i = other.productList().begin();
    ProductRegistry::ProductList::iterator k = productList_.begin();
    for( ; i != e; ++i, ++k) {
      while (k->second.produced()) ++k;
      if (!k->second.merge(i->second, m)) return false;
    }
    return true;
  }
}
