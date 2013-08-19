#ifndef DataFormats_Provenance_ProductRegistry_h
#define DataFormats_Provenance_ProductRegistry_h

/** \class edm::ProductRegistry

     \original author Stefano ARGIRO
     \current author Bill Tanenbaum
     \date 19 Jul 2005
*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"

#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace edm {
  class ProductHolderIndexHelper;

  class ProductRegistry {

  public:
    typedef std::map<BranchKey, BranchDescription> ProductList;

    ProductRegistry();

    // A constructor from the persistent data memebers from another product registry.
    // saves time by not copying the transient components.
    // The constructed registry will be frozen by default.
    explicit ProductRegistry(ProductList const& productList, bool toBeFrozen = true);

    virtual ~ProductRegistry() {}

    typedef std::map<BranchKey, BranchDescription const> ConstProductList;

    void addProduct(BranchDescription const& productdesc, bool iFromListener = false);

    void addLabelAlias(BranchDescription const& productdesc, std::string const& labelAlias, std::string const& instanceAlias);

    void copyProduct(BranchDescription const& productdesc);

    void setFrozen(bool initializeLookupInfo = true);

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
    template<typename T>
    void callForEachBranch(T const& iFunc)  {
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

    ConstProductList& constProductList() {
       //throwIfNotFrozen();
       return transient_.constProductList_;
    }

    boost::shared_ptr<ProductHolderIndexHelper> const& productLookup(BranchType branchType) const;

    // returns the appropriate ProductHolderIndex else ProductHolderIndexInvalid if no BranchID is available
    ProductHolderIndex indexFrom(BranchID const& iID) const;

    bool productProduced(BranchType branchType) const {return transient_.productProduced_[branchType];}
    bool anyProductProduced() const {return transient_.anyProductProduced_;}
    BranchListIndex producedBranchListIndex() const {return transient_.producedBranchListIndex_;}

    void setProducedBranchListIndex(BranchListIndex blix) {
      transient_.producedBranchListIndex_ = blix;
    }

    std::vector<std::string> const& missingDictionaries() const {
      return transient_.missingDictionaries_;
    }

    std::vector<std::string>& missingDictionariesForUpdate() {
      return transient_.missingDictionaries_;
    }

    ProductHolderIndex const& getNextIndexValue(BranchType branchType) const;

    void initializeTransients() {transient_.reset();}

    bool frozen() const {return transient_.frozen_;}

    struct Transients {
      Transients();
      void reset();
      bool frozen_;
      ConstProductList constProductList_;
      // Is at least one (run), (lumi), (event) product produced this process?
      boost::array<bool, NumBranchTypes> productProduced_;
      bool anyProductProduced_;

      boost::shared_ptr<ProductHolderIndexHelper> eventProductLookup_;
      boost::shared_ptr<ProductHolderIndexHelper> lumiProductLookup_;
      boost::shared_ptr<ProductHolderIndexHelper> runProductLookup_;

      ProductHolderIndex eventNextIndexValue_;
      ProductHolderIndex lumiNextIndexValue_;
      ProductHolderIndex runNextIndexValue_;

      std::map<BranchID, ProductHolderIndex> branchIDToIndex_;

      BranchListIndex producedBranchListIndex_;

      std::vector<std::string> missingDictionaries_;
    };

  private:
    void setProductProduced(BranchType branchType) {
      transient_.productProduced_[branchType] = true;
      transient_.anyProductProduced_ = true;
    }

    void freezeIt(bool frozen = true) {transient_.frozen_ = frozen;}

    void updateConstProductRegistry();
    void initializeLookupTables();
    virtual void addCalled(BranchDescription const&, bool iFromListener);
    void throwIfNotFrozen() const;
    void throwIfFrozen() const;

    ProductHolderIndex& nextIndexValue(BranchType branchType);

    ProductList productList_;
    Transients transient_;
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
