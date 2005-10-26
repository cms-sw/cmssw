#ifndef Framework_ProductRegistry_h
#define Framework_ProductRegistry_h

/**
   \file
   Implementation of ProductRegistry

   \author Stefano ARGIRO
   \co-author Bill Tanenbaum
   \version $Id: ProductRegistry.h,v 1.13 2005/10/11 19:28:20 chrjones Exp $
   \date 19 Jul 2005
*/

#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/BranchDescription.h"

namespace edm {

  /**
     \class ProductRegistry ProductRegistry.h "edm/ProductRegistry.h"

     \brief 

     \author Stefano ARGIRO
     \co-author Bill Tanenbaum
     \date 19 Jul 2005
  */
  class ProductRegistry {

  public:
    ProductRegistry() : productList_(), nextID_(0) {}

    virtual ~ProductRegistry() {}
  
    typedef std::map<BranchKey, BranchDescription> ProductList;

    void addProduct(BranchDescription const& productdesc, bool iFromListener=false);

    void copyProduct(BranchDescription const& productdesc);

    void setProductIDs();

    ProductList const& productList() const {return productList_;}
    
    unsigned long nextID() const {return nextID_;}

    void setNextID(unsigned long next) {nextID_ = next;}

  private:
    virtual void addCalled(BranchDescription const&, bool iFromListener);
    ProductList productList_;
    unsigned long nextID_;
  };

  inline
  bool
  operator==(ProductRegistry const& lhs, ProductRegistry const& rhs) {
    return lhs.nextID() == rhs.nextID() && lhs.productList() == rhs.productList();
  }

  inline
  bool
  operator!=(ProductRegistry const& lhs, ProductRegistry const& rhs) {
    return !(lhs == rhs);
  }
} // edm


#endif

