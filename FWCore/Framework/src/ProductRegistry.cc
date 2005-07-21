/**
   \file
   class impl

   \author Stefano ARGIRO
   \version $Id$
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id$";


#include <FWCore/Framework/interface/ProductRegistry.h>

using namespace edm;

void ProductRegistry::addProduct(const ProductDescription& productdesc){
  productList_.push_back(productdesc);
}

const ProductRegistry::ProductList& ProductRegistry::getProductList() const{
  return productList_;
}



// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
