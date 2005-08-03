/**
   \file
   class impl

   \author Stefano ARGIRO
   \co-author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.8 2005/07/30 23:47:00 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.8 2005/07/30 23:47:00 wmtan Exp $";


#include <FWCore/Framework/interface/ProductRegistry.h>
#include <algorithm>

using namespace edm;

void
ProductRegistry::addProduct(ProductDescription& productDesc) {
  productDesc.init();
  productDesc.productID_.id_ = ++nextID_;
  productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
}

void
ProductRegistry::copyProduct(ProductDescription const& productDesc) {
  productDesc.init();
  productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
  if (productDesc.productID_.id_ >= nextID_) {
    nextID_ = productDesc.productID_.id_ + 1;
  }
}


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
