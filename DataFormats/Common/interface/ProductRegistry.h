#ifndef Common_ProductRegistry_h
#define Common_ProductRegistry_h

/**
   \file
   Implementation of ProductRegistry

   \author Stefano ARGIRO
   \co-author Bill Tanenbaum
   \version $Id: ProductRegistry.h,v 1.2 2006/04/17 23:39:20 wmtan Exp $
   \date 19 Jul 2005
*/

#include <map>

#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/BranchDescription.h"

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
    ProductRegistry() : productList_(), nextID_(0), frozen_(false) {}

    virtual ~ProductRegistry() {}

    typedef std::map<BranchKey, BranchDescription> ProductList;

    void addProduct(BranchDescription const& productdesc, bool iFromListener=false);

    void copyProduct(BranchDescription const& productdesc);

    void setProductIDs();

    void setFrozen() const;

    ProductList const& productList() const {
      throwIfNotFrozen();
      return productList_;
    }

    unsigned long nextID() const {return nextID_;}

    void setNextID(unsigned long next) {nextID_ = next;}


    //NOTE: this is not const since we only want items that have non-const access to this class to be 
    // able to call this internal iteration
    template<class T>
    void callForEachBranch(const T& iFunc)  {
      //NOTE: If implementation changes from a map, need to check that iterators are still valid
      // after an insert with the new container, else need to copy the container and iterate over the copy
      for(ProductRegistry::ProductList::const_iterator itEntry=productList_.begin();
          itEntry!=productList_.end(); ++itEntry){
        iFunc(itEntry->second);
      }
    }
    ProductList::size_type size() const {return productList_.size();}

  private:
    virtual void addCalled(BranchDescription const&, bool iFromListener);
    void throwIfNotFrozen() const;
    void throwIfFrozen() const;

    ProductList productList_;
    unsigned long nextID_;
    bool mutable frozen_;
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
