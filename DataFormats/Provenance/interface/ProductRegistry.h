#ifndef DataFormats_Provenance_ProductRegistry_h
#define DataFormats_Provenance_ProductRegistry_h

/**
   \file
   Implementation of ProductRegistry

   \original author Stefano ARGIRO
   \current author Bill Tanenbaum
   \version $Id: ProductRegistry.h,v 1.6 2008/03/24 02:26:02 wmtan Exp $
   \date 19 Jul 2005
*/

#include <map>
#include <set>
#include <iosfwd>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

#include "Reflex/Type.h"

namespace edm {

  /**
     \class ProductRegistry ProductRegistry.h "edm/ProductRegistry.h"

     \brief

     \original author Stefano ARGIRO
     \current author Bill Tanenbaum
     \date 19 Jul 2005
  */
  class ProductRegistry {

  public:
    ProductRegistry();

    virtual ~ProductRegistry() {}

    typedef std::map<BranchKey, BranchDescription> ProductList;

    typedef std::map<BranchKey, ConstBranchDescription> ConstProductList;
    
    // Used for indices to find product IDs by type and process
    typedef std::map<std::string, std::vector<ProductID> > ProcessLookup;
    typedef std::map<std::string, ProcessLookup> TypeLookup;

    void addProduct(BranchDescription const& productdesc, bool iFromListener=false);

    void copyProduct(BranchDescription const& productdesc);

    void setProductIDs();

    void setFrozen() const;

    std::string merge(ProductRegistry const& other,
	std::string const& fileName,
	BranchDescription::MatchMode m);

    ProductList const& productList() const {
      throwIfNotFrozen();
      return productList_;
    }

    ConstProductList const& constProductList() const {
      throwIfNotFrozen();
      return constProductList_;
    }

    unsigned int nextID() const {return nextID_;}

    void setNextID(unsigned int next) {nextID_ = next;}

    unsigned int maxID() const {return maxID_;}

    const TypeLookup& productLookup() const {
      return productLookup_;
    }
    const TypeLookup& elementLookup() const {
      return elementLookup_;
    }

    //NOTE: this is not const since we only want items that have non-const access to this class to be 
    // able to call this internal iteration
    template<class T>
    void callForEachBranch(const T& iFunc)  {
      //NOTE: If implementation changes from a map, need to check that iterators are still valid
      // after an insert with the new container, else need to copy the container and iterate over the copy
      for(ProductRegistry::ProductList::const_iterator itEntry = productList_.begin(),
          itEntryEnd = productList_.end();
          itEntry != itEntryEnd; ++itEntry) {
        iFunc(itEntry->second);
      }
    }
    ProductList::size_type size() const {return productList_.size();}

    void print(std::ostream& os) const;

    void deleteDroppedProducts();

  private:
    
    void initializeTransients() const;
    virtual void addCalled(BranchDescription const&, bool iFromListener);
    void throwIfNotFrozen() const;
    void throwIfFrozen() const;
    void fillElementLookup(const ROOT::Reflex::Type & type,
                           const ProductID& slotNumber,
                           const BranchKey& bk) const;
    
    ProductList productList_;
    unsigned int nextID_;
    mutable unsigned int maxID_;
    mutable bool frozen_;
    mutable ConstProductList constProductList_;
    
    // indices used to quickly find a group in the vector groups_
    // by type, first one by the type of the EDProduct and the
    // second by the type of object contained in a sequence in
    // an EDProduct
    mutable TypeLookup productLookup_; // 1->many
    mutable TypeLookup elementLookup_; // 1->many
    
    // Fix some product ID's to facilitate merging.
    std::map<std::string, unsigned int> fixedProductIDs_;
    std::set<unsigned int> preExistingFixedProductIDs_;
  };

  inline
  bool
  operator==(ProductRegistry const& a, ProductRegistry const& b) {
    return a.nextID() == b.nextID() && a.productList() == b.productList();
  }

  inline
  bool
  operator!=(ProductRegistry const& a, ProductRegistry const& b) {
    return !(a == b);
  }

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductRegistry const& pr) {
    pr.print(os);
    return os;    
  }

} // edm


#endif
