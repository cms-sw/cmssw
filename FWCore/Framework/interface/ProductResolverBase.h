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
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <memory>

#include <string>

namespace edm {
  class ProductProvenanceRetriever;
  class DelayedReader;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class Principal;
  class UnscheduledConfigurator;

  class ProductResolverBase {
  public:

    enum ResolveStatus { ProductFound, ProductNotFound, Ambiguous };

    ProductResolverBase();
    virtual ~ProductResolverBase();

    ProductResolverBase(ProductResolverBase const&) = delete; // Disallow copying and moving
    ProductResolverBase& operator=(ProductResolverBase const&) = delete; // Disallow copying and moving

    ProductData const* resolveProduct(ResolveStatus& resolveStatus,
                                      Principal const& principal,
                                      bool skipCurrentProcess,
                                      SharedResourcesAcquirer* sra,
                                      ModuleCallingContext const* mcc) const {
      return resolveProduct_(resolveStatus, principal, skipCurrentProcess, sra, mcc);
    }

    void resetProductData() { resetProductData_(false); }

    void unsafe_deleteProduct() const {
      const_cast<ProductResolverBase*>(this)->resetProductData_(true);
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

    // Same as moduleLabel except in the case of an AliasProductResolver, in which
    // case it resolves the module which actually produces the product and returns
    // its module label
    std::string const& resolvedModuleLabel() const {return resolvedModuleLabel_();}

    // Retrieves a reference to the product instance name
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}

    // Retrieves a reference to the process name
    std::string const& processName() const {return branchDescription().processName();}

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance const* provenance() const;

    // Retrieves pointer to a class containing the event independent provenance.
    StableProvenance const* stableProvenance() const {return &provenance()->stable();}

    // Initializes the event independent portion of the provenance, plus the process history ID, the product ID, and the provRetriever.
    void setProvenance(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) { setProvenance_(provRetriever, ph, pid); }

    // Initializes the process history.
    void setProcessHistory(ProcessHistory const& ph) { setProcessHistory_(ph); }

    // Write the product to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this ProductResolver.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return provenance()->productID();}

    // Puts the product into the ProductResolver.
    void putProduct(std::unique_ptr<WrapperBase> edp) const {
      putProduct_(std::move(edp));
    }

    // If the product already exists we merge, else will put
    void putOrMergeProduct(std::unique_ptr<WrapperBase> edp) const {
      putOrMergeProduct_(std::move(edp));
    }
    
    virtual void connectTo(ProductResolverBase const&, Principal const*) = 0;
    virtual void setupUnscheduled(UnscheduledConfigurator const&);

  private:
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const = 0;
    virtual bool unscheduledWasNotRun_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual bool productResolved_() const = 0;
    virtual bool productWasDeleted_() const = 0;
    virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const = 0;
    virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const = 0;
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
  operator<<(std::ostream& os, ProductResolverBase const& phb) {
    phb.write(os);
    return os;
  }
}

#endif
