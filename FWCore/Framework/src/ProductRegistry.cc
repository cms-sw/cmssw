/**
   \file
   class impl

   \author Stefano ARGIRO
   \co-author William Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.5 2005/07/27 04:33:48 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.5 2005/07/27 04:33:48 wmtan Exp $";


#include <FWCore/Framework/interface/ProductRegistry.h>
#include <algorithm>

using namespace edm;

void
ProductRegistry::addProduct(ProductDescription& productDesc) {
  productDesc.product_id.id_ = nextID_++;
  productList_.push_back(productDesc);
  sorted_ = false;
}

void
ProductRegistry::copyProduct(ProductDescription const& productDesc) {
  productList_.push_back(productDesc);
  sorted_ = true;
  if (productDesc.product_id.id_ >= nextID_) {
    nextID_ = productDesc.product_id.id_ + 1;
  }
}

void
ProductRegistry::reallySort() {
  std::sort(productList_.begin(), productList_.end());
  sorted_ = true;
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
