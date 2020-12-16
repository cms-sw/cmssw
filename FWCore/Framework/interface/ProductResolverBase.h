#ifndef FWCore_Framework_ProductResolverBase_h
#define FWCore_Framework_ProductResolverBase_h

/*----------------------------------------------------------------------

ProductResolver: Class to handle access to a WrapperBase and its related information.

 [The class was formerly called Group and later ProductHolder]
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <memory>

#include <string>

namespace edm {
  class MergeableRunProductMetadata;
  class ProductProvenanceRetriever;
  class DelayedReader;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class Principal;
  class UnscheduledConfigurator;
  class ServiceToken;

  class ProductResolverBase {
  public:
    class Resolution {
    public:
      static std::uintptr_t constexpr kAmbiguityValue = 0x1;
      static std::uintptr_t constexpr kAmbiguityMask = std::numeric_limits<std::uintptr_t>::max() ^ kAmbiguityValue;
      explicit Resolution(ProductData const* iData) : m_data(iData) {}

      bool isAmbiguous() const { return reinterpret_cast<std::uintptr_t>(m_data) == kAmbiguityValue; }

      ProductData const* data() const {
        return reinterpret_cast<ProductData const*>(kAmbiguityMask & reinterpret_cast<std::uintptr_t>(m_data));
      }

      static Resolution makeAmbiguous() { return Resolution(reinterpret_cast<ProductData const*>(kAmbiguityValue)); }

    private:
      ProductData const* m_data;
    };

    ProductResolverBase();
    virtual ~ProductResolverBase();

    ProductResolverBase(ProductResolverBase const&) = delete;             // Disallow copying and moving
    ProductResolverBase& operator=(ProductResolverBase const&) = delete;  // Disallow copying and moving

    Resolution resolveProduct(Principal const& principal,
                              bool skipCurrentProcess,
                              SharedResourcesAcquirer* sra,
                              ModuleCallingContext const* mcc) const {
      return resolveProduct_(principal, skipCurrentProcess, sra, mcc);
    }

    /** oDataFetchedIsValid is allowed to be nullptr in which case no value will be assigned
     */
    void prefetchAsync(WaitingTaskHolder waitTask,
                       Principal const& principal,
                       bool skipCurrentProcess,
                       ServiceToken const& token,
                       SharedResourcesAcquirer* sra,
                       ModuleCallingContext const* mcc) const {
      return prefetchAsync_(waitTask, principal, skipCurrentProcess, token, sra, mcc);
    }

    void retrieveAndMerge(Principal const& principal,
                          MergeableRunProductMetadata const* mergeableRunProductMetadata) const {
      retrieveAndMerge_(principal, mergeableRunProductMetadata);
    }
    void resetProductData() { resetProductData_(false); }

    void unsafe_deleteProduct() const { const_cast<ProductResolverBase*>(this)->resetProductData_(true); }

    // product is not available (dropped or never created)
    bool productUnavailable() const { return productUnavailable_(); }

    // returns true if resolveProduct was already called for this product
    bool productResolved() const { return productResolved_(); }

    // provenance is currently available
    bool provenanceAvailable() const;

    // Only returns true if the module is unscheduled and was not run
    //   all other cases return false
    bool unscheduledWasNotRun() const { return unscheduledWasNotRun_(); }

    // Product was deleted early in order to save memory
    bool productWasDeleted() const { return productWasDeleted_(); }

    bool productWasFetchedAndIsValid(bool iSkipCurrentProcess) const {
      return productWasFetchedAndIsValid_(iSkipCurrentProcess);
    }

    // Retrieves pointer to the per event(lumi)(run) provenance.
    ProductProvenance const* productProvenancePtr() const { return productProvenancePtr_(); }

    // Retrieves a reference to the event independent provenance.
    BranchDescription const& branchDescription() const { return branchDescription_(); }

    // Retrieves a reference to the event independent provenance.
    bool singleProduct() const { return singleProduct_(); }

    // Sets the pointer to the event independent provenance.
    void resetBranchDescription(std::shared_ptr<BranchDescription const> bd) { resetBranchDescription_(bd); }

    // Retrieves a reference to the module label.
    std::string const& moduleLabel() const { return branchDescription().moduleLabel(); }

    // Same as moduleLabel except in the case of an AliasProductResolver, in which
    // case it resolves the module which actually produces the product and returns
    // its module label
    std::string const& resolvedModuleLabel() const { return resolvedModuleLabel_(); }

    // Retrieves a reference to the product instance name
    std::string const& productInstanceName() const { return branchDescription().productInstanceName(); }

    // Retrieves a reference to the process name
    std::string const& processName() const { return branchDescription().processName(); }

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance const* provenance() const;

    // Retrieves pointer to a class containing the event independent provenance.
    StableProvenance const* stableProvenance() const { return &provenance()->stable(); }

    // Initialize the mechanism to retrieve per event provenance
    void setProductProvenanceRetriever(ProductProvenanceRetriever const* provRetriever) {
      setProductProvenanceRetriever_(provRetriever);
    }

    // Initializes the ProductID
    void setProductID(ProductID const& pid) { setProductID_(pid); }

    // Initializes the portion of the provenance related to mergeable run products.
    void setMergeableRunProductMetadata(MergeableRunProductMetadata const* mrpm) {
      setMergeableRunProductMetadata_(mrpm);
    }

    // Write the product to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this ProductResolver.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const { return provenance()->productID(); }

    // Puts the product into the ProductResolver.
    void putProduct(std::unique_ptr<WrapperBase> edp) const { putProduct_(std::move(edp)); }

    // If the product already exists we merge, else will put
    void putOrMergeProduct(std::unique_ptr<WrapperBase> edp,
                           MergeableRunProductMetadata const* mergeableRunProductMetadata = nullptr) const {
      putOrMergeProduct_(std::move(edp), mergeableRunProductMetadata);
    }

    virtual void connectTo(ProductResolverBase const&, Principal const*) = 0;
    virtual void setupUnscheduled(UnscheduledConfigurator const&);

  private:
    virtual Resolution resolveProduct_(Principal const& principal,
                                       bool skipCurrentProcess,
                                       SharedResourcesAcquirer* sra,
                                       ModuleCallingContext const* mcc) const = 0;
    virtual void prefetchAsync_(WaitingTaskHolder waitTask,
                                Principal const& principal,
                                bool skipCurrentProcess,
                                ServiceToken const& token,
                                SharedResourcesAcquirer* sra,
                                ModuleCallingContext const* mcc) const = 0;

    virtual void retrieveAndMerge_(Principal const& principal,
                                   MergeableRunProductMetadata const* mergeableRunProductMetadata) const;

    virtual bool unscheduledWasNotRun_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual bool productResolved_() const = 0;
    virtual bool productWasDeleted_() const = 0;
    virtual bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const = 0;

    virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const = 0;
    virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> edp,
                                    MergeableRunProductMetadata const* mergeableRunProductMetadata) const = 0;
    virtual BranchDescription const& branchDescription_() const = 0;
    virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) = 0;
    virtual Provenance const* provenance_() const = 0;
    virtual std::string const& resolvedModuleLabel_() const = 0;
    virtual void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) = 0;
    virtual void setProductID_(ProductID const& pid) = 0;
    virtual void setMergeableRunProductMetadata_(MergeableRunProductMetadata const*);
    virtual ProductProvenance const* productProvenancePtr_() const = 0;
    virtual void resetProductData_(bool deleteEarly) = 0;
    virtual bool singleProduct_() const = 0;
  };

  inline std::ostream& operator<<(std::ostream& os, ProductResolverBase const& phb) {
    phb.write(os);
    return os;
  }
}  // namespace edm

#endif
