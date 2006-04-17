/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.1 2006/02/08 00:44:23 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.1 2006/02/08 00:44:23 wmtan Exp $";


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
    if (productDesc.productID_.id_ >= nextID_) {
      nextID_ = productDesc.productID_.id_ + 1;
    }
  }
  
  void
  ProductRegistry::setProductIDs() {
    throwIfFrozen();
    for (ProductList::iterator it = productList_.begin(); it != productList_.end(); ++it) {
       if (it->second.productID_.id_ == 0) {
          it->second.productID_.id_ = ++nextID_;
       }
    }
    frozen_ = true;
  }
  
  void
  ProductRegistry::setFrozen() const {
    if (frozen_) return;
    for (ProductList::const_iterator it = productList_.begin(); it != productList_.end(); ++it) {
      if (it->second.productID_.id_ == 0) {
       throw cms::Exception("ProductRegistry", "setFrozen")
          << "cannot read the ProductRegistry because it is not yet frozen.";
      }
    }
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
    if (!frozen_) {
// FIX THIS: Temporarily disabled until the EDProducer callback problem is solved.
//      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
//            << "cannot read the ProductRegistry because it is not yet frozen";
    }
  }
  
  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
  }
}
