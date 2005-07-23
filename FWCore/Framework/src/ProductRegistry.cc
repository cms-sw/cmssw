/**
   \file
   class impl

   \author Stefano ARGIRO
   \version $Id: ProductRegistry.cc,v 1.1 2005/07/21 20:47:28 argiro Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.1 2005/07/21 20:47:28 argiro Exp $";


#include <FWCore/Framework/interface/ProductRegistry.h>

using namespace edm;

void ProductRegistry::addProduct(ProductDescription& productDesc) {
  productDesc.product_id.id_ = productList_.size();
  productList_.push_back(productDesc);
}

const ProductRegistry::ProductList& ProductRegistry::productList() const {
  return productList_;
}



// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
