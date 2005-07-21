/**
   \file
   Implementation of ProductRegistry

   \author Stefano ARGIRO
   \version $Id$
   \date 19 Jul 2005
*/

#ifndef _edm_ProductRegistry_h_
#define _edm_ProductRegistry_h_

static const char CVSId_edm_ProductRegistry[] = 
"$Id$";

#include <FWCore/Framework/interface/ProductDescription.h>
#include <list>
namespace edm {

  

  /**
     \class ProductRegistry ProductRegistry.h "edm/ProductRegistry.h"

     \brief 

     \author Stefano ARGIRO
     \date 19 Jul 2005
  */
  class ProductRegistry {

  public:
  
    typedef std::list<ProductDescription> ProductList;

    void addProduct(const ProductDescription& productdesc);

    const ProductList& getProductList() const;
    
    
  private:
    std::list<ProductDescription> productList_;
    
  };
} // edm


#endif // _edm_ProductRegistry_h_

