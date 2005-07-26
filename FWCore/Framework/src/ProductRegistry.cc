/**
   \file
   class impl

   \author Stefano ARGIRO
   \co-author William Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.2 2005/07/23 05:10:32 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.2 2005/07/23 05:10:32 wmtan Exp $";


#include <FWCore/Framework/interface/ProductRegistry.h>
#include <algorithms>

using namespace edm;

void
ProductRegistry::addProduct(ProductDescription& productDesc) {
  productDesc.product_id.id_ = productList_.size();
  productList_.push_back(productDesc);
  sorted_ = false;
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
