/**
   \file
   class impl

   \author Stefano ARGIRO
   \co-author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.12 2005/10/07 19:05:18 chrjones Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.12 2005/10/07 19:05:18 chrjones Exp $";


#include <FWCore/Framework/interface/ProductRegistry.h>
#include <algorithm>

using namespace edm;

void
ProductRegistry::addProduct(BranchDescription const& productDesc, bool fromListener) {
  productDesc.init();
  productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
  addCalled(productDesc,fromListener);
}

void
ProductRegistry::copyProduct(BranchDescription const& productDesc) {
  productDesc.init();
  productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
  if (productDesc.productID_.id_ >= nextID_) {
    nextID_ = productDesc.productID_.id_ + 1;
  }
}

void
ProductRegistry::setProductIDs() {
  for (ProductList::iterator it = productList_.begin(); it != productList_.end(); ++it) {
     if (it->second.productID_.id_ == 0) {
        it->second.productID_.id_ = ++nextID_;
     }
  }
}

void
ProductRegistry::addCalled(BranchDescription const&, bool) {
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
