/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.15 2006/01/24 16:46:03 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.15 2006/01/24 16:46:03 wmtan Exp $";


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
    setFrozen();
  }
  
  void
  ProductRegistry::throwIfFrozen() const {
    if (frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
            << "cannot modify ProductRegistry becauseit is frozen";
    }
  }
  
  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
  }
}
