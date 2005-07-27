/**
   \file
   Implementation of ProductRegistry

   \author Stefano ARGIRO
   \version $Id: ProductRegistry.h,v 1.4 2005/07/26 23:03:29 wmtan Exp $
   \date 19 Jul 2005
*/

#ifndef Framework_ProductRegistry_h
#define Framework_ProductRegistry_h

static const char CVSId_edm_ProductRegistry[] = 
"$Id: ProductRegistry.h,v 1.4 2005/07/26 23:03:29 wmtan Exp $";

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
    ProductRegistry() : productList_(), nextID_(0), sorted_(false) {}

    ~ProductRegistry() {}
  
    typedef std::vector<ProductDescription> ProductList;

    void addProduct(ProductDescription& productdesc);

    ProductList const& productList() const {return productList_;}
    
    void sort() {
      if(sorted_) return;
      reallySort();
    }

  private:
    void reallySort() ;

    ProductList productList_;
    unsigned long nextID_;
    bool sorted_;
  };
} // edm


#endif // _edm_ProductRegistry_h_

