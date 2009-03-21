#ifndef DataFormats_Provenance_ProductRegistry_h
#define DataFormats_Provenance_ProductRegistry_h

/**
   \file
   Implementation of ProductRegistry

   \original author Stefano ARGIRO
   \current author Bill Tanenbaum
   \date 19 Jul 2005
*/

#include <map>
#include <set>
#include <iosfwd>
#include <string>
#include <vector>

#include "boost/array.hpp"

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Transient.h"
#include "FWCore/Utilities/interface/TypeID.h"

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
    typedef std::map<BranchKey, BranchDescription> ProductList;

    ProductRegistry();

    // A constructor from the persistent data memebers from another product registry.
    // saves time by not copying the transient components.
    // The constructed registry will be frozen.
    explicit ProductRegistry(ProductList const& productList);

    virtual ~ProductRegistry() {}


    typedef std::map<BranchKey, ConstBranchDescription> ConstProductList;
    
    // Used for indices to find branch IDs by type and process
    typedef std::map<std::string, std::vector<BranchID> > ProcessLookup;
    typedef std::map<edm::TypeID, ProcessLookup> TypeLookup;

    void addProduct(BranchDescription const& productdesc, bool iFromListener=false);

    void copyProduct(BranchDescription const& productdesc);

    void setFrozen() const;

    std::string merge(ProductRegistry const& other,
	std::string const& fileName,
	BranchDescription::MatchMode parametersMustMatch = BranchDescription::Permissive,
	BranchDescription::MatchMode branchesMustMatch = BranchDescription::Permissive);

    void updateFromInput(ProductList const& other);

    void updateFromInput(std::vector<BranchDescription> const& other);

    ProductList const& productList() const {
      //throwIfNotFrozen();
      return productList_;
    }

    ProductList& productListUpdator() {
      throwIfFrozen();
      return productList_;
    }

    // Return all the branch names currently known to *this.  This
    // does a return-by-value of the vector so that it may be used in
    // a colon-initialization list.
    std::vector<std::string> allBranchNames() const;

    // Return pointers to (const) BranchDescriptions for all the
    // BranchDescriptions known to *this.  This does a
    // return-by-value of the vector so that it may be used in a
    // colon-initialization list.
    std::vector<BranchDescription const*> allBranchDescriptions() const;
     
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

    bool anyProducts(BranchType const brType) const;

    ConstProductList & constProductList() const {
	 //throwIfNotFrozen();
       return transients_.get().constProductList_;
    }

    TypeLookup & productLookup() const {return transients_.get().productLookup_;}

    TypeLookup & elementLookup() const {return transients_.get().elementLookup_;}

    struct Transients {
      Transients();
      bool frozen_;
      ConstProductList constProductList_; 
      // Is at least one (run), (lumi), (event) product produced this process?
      boost::array<bool, NumBranchTypes> productProduced_;

      // indices used to quickly find a group in the vector groups_
      // by type, first one by the type of the EDProduct and the
      // second by the type of object contained in a sequence in
      // an EDProduct
      TypeLookup productLookup_;
      TypeLookup elementLookup_;
    };

    bool productProduced(BranchType branchType) const {return transients_.get().productProduced_[branchType];}

  private:
    void setProductProduced(BranchType branchType) const {transients_.get().productProduced_[branchType] = true;}

    bool & frozen() const {return transients_.get().frozen_;}
    
    void initializeTransients() const;
    virtual void addCalled(BranchDescription const&, bool iFromListener);
    void throwIfNotFrozen() const;
    void throwIfFrozen() const;
    void fillElementLookup(const Reflex::Type & type,
                           const BranchID& slotNumber,
                           const BranchKey& bk) const;
    
    ProductList productList_;
    mutable Transient<Transients> transients_;
    
  };

  inline
  bool
  operator==(ProductRegistry const& a, ProductRegistry const& b) {
    return a.productList() == b.productList();
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
