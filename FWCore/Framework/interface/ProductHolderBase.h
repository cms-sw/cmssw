#ifndef FWCore_Framework_ProductHolderBase_h
#define FWCore_Framework_ProductHolderBase_h

/*----------------------------------------------------------------------

ProductHolder: A collection of information related to a single WrapperBase or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <memory>

#include <string>

namespace edm {
  class ProductProvenanceRetriever;
  class DelayedReader;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class Principal;
  class AliasProductHolder;
  class DataManagingProductHolder;
  class ParentProcessProductHolder;

  class ProductHolderBase {
  public:

    friend class AliasProductHolder;
    friend class DataManagingProductHolder;
    friend class ParentProcessProductHolder;
    
    enum ResolveStatus { ProductFound, ProductNotFound, Ambiguous };

    ProductHolderBase();
    virtual ~ProductHolderBase();

    ProductHolderBase(ProductHolderBase const&) = delete; // Disallow copying and moving
    ProductHolderBase& operator=(ProductHolderBase const&) = delete; // Disallow copying and moving

    ProductData const* resolveProduct(ResolveStatus& resolveStatus,
                                      Principal const& principal,
                                      bool skipCurrentProcess,
                                      SharedResourcesAcquirer* sra,
                                      ModuleCallingContext const* mcc) const {
      return resolveProduct_(resolveStatus, principal, skipCurrentProcess, sra, mcc);
    }

    void resetProductData() { resetProductData_(false); }

    void unsafe_deleteProduct() const {
      const_cast<ProductHolderBase*>(this)->resetProductData_(true);
    }
    
    // product is not available (dropped or never created)
    bool productUnavailable() const {return productUnavailable_();}
    
    // returns true if resolveProduct was already called for this product
    bool productResolved() const { return productResolved_(); }

    // provenance is currently available
    bool provenanceAvailable() const;

    // Only returns true if the module is unscheduled and was not run
    //   all other cases return false
    bool unscheduledWasNotRun() const {return unscheduledWasNotRun_();}
    
    // Product was deleted early in order to save memory
    bool productWasDeleted() const {return productWasDeleted_();}

    // Retrieves pointer to the per event(lumi)(run) provenance.
    ProductProvenance const* productProvenancePtr() const { return productProvenancePtr_(); }

    // Retrieves a reference to the event independent provenance.
    BranchDescription const& branchDescription() const {return branchDescription_();}

    // Retrieves a reference to the event independent provenance.
    bool singleProduct() const {return singleProduct_();}

    // Sets the pointer to the event independent provenance.
    void resetBranchDescription(std::shared_ptr<BranchDescription const> bd) {resetBranchDescription_(bd);}

    // Retrieves a reference to the module label.
    std::string const& moduleLabel() const {return branchDescription().moduleLabel();}

    // Same as moduleLabel except in the case of an AliasProductHolder, in which
    // case it resolves the module which actually produces the product and returns
    // its module label
    std::string const& resolvedModuleLabel() const {return resolvedModuleLabel_();}

    // Retrieves a reference to the product instance name
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}

    // Retrieves a reference to the process name
    std::string const& processName() const {return branchDescription().processName();}

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance const* provenance() const;

    // Initializes the event independent portion of the provenance, plus the process history ID, the product ID, and the provRetriever.
    void setProvenance(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) { setProvenance_(provRetriever, ph, pid); }

    // Initializes the process history.
    void setProcessHistory(ProcessHistory const& ph) { setProcessHistory_(ph); }

    // Write the product to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this ProductHolder.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return provenance()->productID();}

    // Puts the product into the ProductHolder.
    void putProduct(std::unique_ptr<WrapperBase> edp) const {
      putProduct_(std::move(edp));
    }

    // This returns true if it will be put, false if it will be merged
    bool putOrMergeProduct() const {
      return putOrMergeProduct_();
    }

    // merges the product with the pre-existing product
    void mergeProduct(std::unique_ptr<WrapperBase> edp) const {
      mergeProduct_(std::move(edp));
    }

    void reallyCheckType(WrapperBase const& prod) const;

    void checkType(WrapperBase const& prod) const {
      checkType_(prod);
    }

    void swap(ProductHolderBase& rhs) {swap_(rhs);}

    void throwProductDeletedException() const;

    virtual void connectTo(ProductHolderBase const&, Principal const*) = 0;

  private:
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const = 0;
    virtual void swap_(ProductHolderBase& rhs) = 0;
    virtual bool unscheduledWasNotRun_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual bool productResolved_() const = 0;
    virtual bool productWasDeleted_() const = 0;
    virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const = 0;
    virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const = 0;
    virtual bool putOrMergeProduct_() const = 0;
    virtual void checkType_(WrapperBase const& prod) const = 0;
    virtual BranchDescription const& branchDescription_() const = 0;
    virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) = 0;
    virtual Provenance const* provenance_() const = 0;
    virtual std::string const& resolvedModuleLabel_() const = 0;
    virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) = 0;
    virtual void setProcessHistory_(ProcessHistory const& ph) = 0;
    virtual ProductProvenance const* productProvenancePtr_() const = 0;
    virtual void resetProductData_(bool deleteEarly) = 0;
    virtual bool singleProduct_() const = 0;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductHolderBase const& phb) {
    phb.write(os);
    return os;
  }
}

#endif
