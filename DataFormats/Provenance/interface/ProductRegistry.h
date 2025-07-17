#ifndef DataFormats_Provenance_ProductRegistry_h
#define DataFormats_Provenance_ProductRegistry_h

/** \class edm::ProductRegistry

     \original author Stefano ARGIRO
     \current author Bill Tanenbaum
     \date 19 Jul 2005
*/

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <array>
#include <memory>

#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace edm {
  class SignallingProductRegistryFiller;
  class ProductResolverIndexHelper;
  class TypeID;

  class ProductRegistry {
  public:
    friend class SignallingProductRegistryFiller;

    typedef std::map<BranchKey, ProductDescription> ProductList;

    ProductRegistry() = default;
    ProductRegistry(const ProductRegistry&) = default;
    ProductRegistry(ProductRegistry&&) = default;
    ProductRegistry& operator=(ProductRegistry&&) = default;
    // A constructor from the persistent data members from another product registry.
    // saves time by not copying the transient components.
    // The constructed registry will be frozen by default.
    explicit ProductRegistry(ProductList const& productList, bool toBeFrozen = true);

    ~ProductRegistry() = default;

    typedef std::map<BranchKey, ProductDescription const> ConstProductList;

    void copyProduct(ProductDescription const& productdesc);

    void setFrozen(bool initializeLookupInfo = true);

    void setFrozen(std::set<TypeID> const& productTypesConsumed,
                   std::set<TypeID> const& elementTypesConsumed,
                   std::string const& processName);

    void setUnscheduledProducts(std::set<std::string> const& unscheduledLabels);

    std::string merge(ProductRegistry const& other,
                      std::string const& fileName,
                      ProductDescription::MatchMode branchesMustMatch = ProductDescription::Permissive);

    void updateFromInput(ProductList const& other);

    void updateFromInput(std::vector<ProductDescription> const& other);

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

    // Return pointers to (const) ProductDescriptions for all the
    // ProductDescriptions known to *this.  This does a
    // return-by-value of the vector so that it may be used in a
    // colon-initialization list.
    std::vector<ProductDescription const*> allProductDescriptions() const;

    ProductList::size_type size() const { return productList_.size(); }

    //If the value differs between two versions of the main registry then
    // one must update any related meta data
    using CacheID = ProductList::size_type;
    CacheID cacheIdentifier() const { return size(); }

    void print(std::ostream& os) const;

    bool anyProducts(BranchType const brType) const;

    std::shared_ptr<ProductResolverIndexHelper const> productLookup(BranchType branchType) const;

    // returns the appropriate ProductResolverIndex else ProductResolverIndexInvalid if no BranchID is available
    ProductResolverIndex indexFrom(BranchID const& iID) const;

    bool productProduced(BranchType branchType) const { return transient_.productProduced_[branchType]; }
    bool anyProductProduced() const { return transient_.anyProductProduced_; }

    // Looks if a (type, moduleLabel, productInstanceName) is an alias to some other branch
    //
    // Can return multiple modules if kindOfType is ELEMENT_TYPE (i.e.
    // the product is consumed via edm::View) and there is ambiguity
    // (in which case actual Event::get() would eventually lead to an
    // exception). In that case all possible modules whose product
    // could be consumed are returned.
    std::vector<std::string> aliasToModules(KindOfType kindOfType,
                                            TypeID const& type,
                                            std::string_view moduleLabel,
                                            std::string_view productInstanceName) const;

    ProductResolverIndex const& getNextIndexValue(BranchType branchType) const;

    void initializeTransients() { transient_.reset(); }

    bool frozen() const { return transient_.frozen_; }

    struct Transients {
      Transients();
      void reset();

      bool frozen_;
      // Is at least one (run), (lumi), (event) persistent product produced this process?
      std::array<bool, NumBranchTypes> productProduced_;
      bool anyProductProduced_;

      std::array<std::shared_ptr<const ProductResolverIndexHelper>, NumBranchTypes> productLookups_;

      std::array<ProductResolverIndex, NumBranchTypes> nextIndexValues_;

      std::map<BranchID, ProductResolverIndex> branchIDToIndex_;

      enum { kKind, kType, kModuleLabel, kProductInstanceName, kAliasForModuleLabel };
      using AliasToOriginalVector = std::vector<std::tuple<KindOfType, TypeID, std::string, std::string, std::string>>;
      AliasToOriginalVector aliasToOriginal_;
    };

  private:
    //The following three routines are only called by SignallingProductRegistryFiller
    void addProduct_(ProductDescription const& productdesc);

    ProductDescription const& addLabelAlias_(ProductDescription const& productdesc,
                                             std::string const& labelAlias,
                                             std::string const& instanceAlias);

    // triggers callbacks for modules watching registration
    template <typename F>
    void addFromInput_(edm::ProductRegistry const& iReg, F&& iCallback) {
      throwIfFrozen();
      for (auto const& prod : iReg.productList_) {
        ProductList::iterator iter = productList_.find(prod.first);
        if (iter == productList_.end()) {
          productList_.insert(std::make_pair(prod.first, prod.second));
          iCallback(prod.second);
        } else {
          assert(combinable(iter->second, prod.second));
          iter->second.merge(prod.second);
        }
      }
    }

  private:
    void setProductProduced(BranchType branchType) {
      transient_.productProduced_[branchType] = true;
      transient_.anyProductProduced_ = true;
    }

    void freezeIt(bool frozen = true) { transient_.frozen_ = frozen; }

    void initializeLookupTables(std::set<TypeID> const* productTypesConsumed,
                                std::set<TypeID> const* elementTypesConsumed,
                                std::string const* processName);
    void addElementTypesForAliases(std::set<TypeID> const* elementTypesConsumed,
                                   std::map<TypeID, TypeID> const& containedTypeMap,
                                   std::map<TypeID, std::vector<TypeID>> const& containedTypeToBaseTypesMap);

    void checkDictionariesOfConsumedTypes(std::set<TypeID> const* productTypesConsumed,
                                          std::set<TypeID> const* elementTypesConsumed,
                                          std::map<TypeID, TypeID> const& containedTypeMap,
                                          std::map<TypeID, std::vector<TypeID>>& containedTypeToBaseTypesMap);

    void checkForDuplicateProcessName(ProductDescription const& desc, std::string const* processName) const;

    void throwIfNotFrozen() const;
    void throwIfFrozen() const;

    ProductResolverIndex& nextIndexValue(BranchType branchType);

    ProductList productList_;
    Transients transient_;
  };

  inline bool operator==(ProductRegistry const& a, ProductRegistry const& b) {
    return a.productList() == b.productList();
  }

  inline bool operator!=(ProductRegistry const& a, ProductRegistry const& b) { return !(a == b); }

  inline std::ostream& operator<<(std::ostream& os, ProductRegistry const& pr) {
    pr.print(os);
    return os;
  }

}  // namespace edm

#endif
