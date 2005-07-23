/**
   \file
   Implementation of ProductRegistry

   \author Stefano ARGIRO
   \version $Id: ProductRegistry.h,v 1.2 2005/07/22 23:47:15 wmtan Exp $
   \date 19 Jul 2005
*/

#ifndef Framework_ProductRegistry_h
#define Framework_ProductRegistry_h

static const char CVSId_edm_ProductRegistry[] = 
"$Id: ProductRegistry.h,v 1.2 2005/07/22 23:47:15 wmtan Exp $";

#include <FWCore/Framework/interface/ProductDescription.h>
#include <vector>
namespace edm {

  /**
     \class ProductRegistry ProductRegistry.h "edm/ProductRegistry.h"

     \brief 

     \author Stefano ARGIRO
     \date 19 Jul 2005
  */
  class ProductRegistry {

  public:
  
    typedef std::vector<ProductDescription> ProductList;

    void addProduct(ProductDescription& productdesc);

    ProductList const& productList() const;
    
  private:
    std::vector<ProductDescription> productList_;
    
  };
} // edm


#endif // _edm_ProductRegistry_h_

