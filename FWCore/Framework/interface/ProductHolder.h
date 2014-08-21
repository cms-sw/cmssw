#ifndef FWCore_Framework_ProductHolder_h
#define FWCore_Framework_ProductHolder_h

/*----------------------------------------------------------------------

ProductHolder: A collection of information related to a single EDProduct or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProduct.h"
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
  class Principal;

  class ProductHolderBase {
  public:

    enum ResolveStatus { ProductFound, ProductNotFound, Ambiguous };

    ProductHolderBase();
    virtual ~ProductHolderBase();

    ProductHolderBase(ProductHolderBase const&) = delete; // Disallow copying and moving
    ProductHolderBase& operator=(ProductHolderBase const&) = delete; // Disallow copying and moving

    ProductData const& productData() const {
      return getProductData();
    }

    ProductData& productData() {
      return getProductData();
    }

    ProductData const* resolveProduct(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                      ModuleCallingContext const* mcc) const {
      return resolveProduct_(resolveStatus, skipCurrentProcess, mcc);
    }

    void resetStatus () {
      resetStatus_();
    }

    void setProductDeleted () {
      setProductDeleted_();
    }

    void resetProductData() { resetProductData_(); }

    void deleteProduct() {
      getProductData().resetProductData();
      setProductDeleted_();
    }
    
    // product is not available (dropped or never created)
    bool productUnavailable() const {return productUnavailable_();}

    // provenance is currently available
    bool provenanceAvailable() const;

    // Scheduled for on demand production
    bool onDemand() const {return onDemand_();}
    
    // Product was deleted early in order to save memory
    bool productWasDeleted() const {return productWasDeleted_();}

    // Retrieves a pointer to the wrapper of the product.
    EDProduct* product() const { return getProductData().wrapper_.get(); }

    // Retrieves pointer to the per event(lumi)(run) provenance.
    ProductProvenance* productProvenancePtr() const { return productProvenancePtr_(); }

    // Sets the the per event(lumi)(run) provenance.
    void setProductProvenance(ProductProvenance const& prov) const;

    // Retrieves a reference to the event independent provenance.
    BranchDescription const& branchDescription() const {return branchDescription_();}

    // Retrieves a reference to the event independent provenance.
    bool singleProduct() const {return singleProduct_();}

    void setPrincipal(Principal* principal) { setPrincipal_(principal); }

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
    Provenance* provenance() const;

    // Initializes the event independent portion of the provenance, plus the process history ID, the product ID, and the provRetriever.
    void setProvenance(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) { setProvenance_(provRetriever, ph, pid); }

    // Initializes the process history.
    void setProcessHistory(ProcessHistory const& ph) { setProcessHistory_(ph); }

    // Write the product to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this ProductHolder.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return getProductData().prov_.productID();}

    // Puts the product and its per event(lumi)(run) provenance into the ProductHolder.
    void putProduct(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) {
      putProduct_(std::move(edp), productProvenance);
    }

    // Puts the product into the ProductHolder.
    void putProduct(std::unique_ptr<EDProduct> edp) const {
      putProduct_(std::move(edp));
    }

    // This returns true if it will be put, false if it will be merged
    bool putOrMergeProduct() const {
      return putOrMergeProduct_();
    }

    // merges the product with the pre-existing product
    void mergeProduct(std::unique_ptr<EDProduct> edp, ProductProvenance& productProvenance) {
      mergeProduct_(std::move(edp), productProvenance);
    }

    void mergeProduct(std::unique_ptr<EDProduct> edp) const {
      mergeProduct_(std::move(edp));
    }

    // Merges two instances of the product.
    void mergeTheProduct(std::unique_ptr<EDProduct> edp) const;

    void reallyCheckType(EDProduct const& prod) const;

    void checkType(EDProduct const& prod) const {
      checkType_(prod);
    }

    void swap(ProductHolderBase& rhs) {swap_(rhs);}

    void throwProductDeletedException() const;

  private:
    virtual ProductData const& getProductData() const = 0;
    virtual ProductData& getProductData() = 0;
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                               ModuleCallingContext const* mcc) const = 0;
    virtual void swap_(ProductHolderBase& rhs) = 0;
    virtual bool onDemand_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual bool productWasDeleted_() const = 0;
    virtual void putProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) = 0;
    virtual void putProduct_(std::unique_ptr<EDProduct> edp) const = 0;
    virtual void mergeProduct_(std::unique_ptr<EDProduct>  edp, ProductProvenance& productProvenance) = 0;
    virtual void mergeProduct_(std::unique_ptr<EDProduct> edp) const = 0;
    virtual bool putOrMergeProduct_() const = 0;
    virtual void checkType_(EDProduct const& prod) const = 0;
    virtual void resetStatus_() = 0;
    virtual void setProductDeleted_() = 0;
    virtual BranchDescription const& branchDescription_() const = 0;
    virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) = 0;
    virtual std::string const& resolvedModuleLabel_() const = 0;
    virtual void setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) = 0;
    virtual void setProcessHistory_(ProcessHistory const& ph) = 0;
    virtual ProductProvenance* productProvenancePtr_() const = 0;
    virtual void resetProductData_() = 0;
    virtual bool singleProduct_() const = 0;
    virtual void setPrincipal_(Principal* principal) = 0;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductHolderBase const& phb) {
    phb.write(os);
    return os;
  }

  class InputProductHolder : public ProductHolderBase {
    public:
    explicit InputProductHolder(std::shared_ptr<BranchDescription const> bd, Principal* principal) :
        ProductHolderBase(), productData_(bd), productIsUnavailable_(false),
        productHasBeenDeleted_(false), principal_(principal) {}
      virtual ~InputProductHolder();

      // The following is const because we can add an EDProduct to the
      // cache after creation of the ProductHolder, without changing the meaning
      // of the ProductHolder.
      void setProduct(std::unique_ptr<EDProduct> prod) const;
      bool productIsUnavailable() const {return productIsUnavailable_;}
      void setProductUnavailable() const {productIsUnavailable_ = true;}

    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        InputProductHolder& other = dynamic_cast<InputProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(productIsUnavailable_, other.productIsUnavailable_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void putProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) override;
      virtual void putProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance& productProvenance) override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(EDProduct const&) const override {}
      virtual void resetStatus_() override {productIsUnavailable_ = false;
        productHasBeenDeleted_=false;}
      virtual bool onDemand_() const override {return false;}
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override {return productHasBeenDeleted_;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual void setProductDeleted_() override {productHasBeenDeleted_ = true;}
      virtual BranchDescription const& branchDescription_() const override {return *productData().branchDescription();}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {productData().resetBranchDescription(bd);}
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;
      virtual void setPrincipal_(Principal* principal) override;

      ProductData productData_;
      mutable bool productIsUnavailable_;
      mutable bool productHasBeenDeleted_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(InputProductHolder& a, InputProductHolder& b) {
    a.swap(b);
  }

  class ProducedProductHolder : public ProductHolderBase {
    public:
    enum ProductStatus {
      Present = 0,
      NotRun = 3,
      NotCompleted = 4,
      NotPut = 5,
      UnscheduledNotRun = 6,
      ProductDeleted =7,
      Uninitialized = 0xff
    };
      ProducedProductHolder() : ProductHolderBase() {}
      virtual ~ProducedProductHolder();
      void producerStarted();
      void producerCompleted();
      ProductStatus& status() const {return status_();}
    private:
      virtual void putProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) override;
      virtual void putProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance& productProvenance) override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(EDProduct const& prod) const override {
        reallyCheckType(prod);
      }
      virtual ProductStatus& status_() const = 0;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual void setProductDeleted_() override;
      virtual BranchDescription const& branchDescription_() const override {return *productData().branchDescription();}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {productData().resetBranchDescription(bd);}
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;
      virtual void setPrincipal_(Principal* principal) override;
  };

  class ScheduledProductHolder : public ProducedProductHolder {
    public:
      explicit ScheduledProductHolder(std::shared_ptr<BranchDescription const> bd) : ProducedProductHolder(), productData_(bd), theStatus_(NotRun) {}
      virtual ~ScheduledProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        ScheduledProductHolder& other = dynamic_cast<ScheduledProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void resetStatus_() override {theStatus_ = NotRun;}
      virtual bool onDemand_() const override {return false;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual ProductStatus& status_() const override {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
  };

  // Free swap function
  inline void swap(ScheduledProductHolder& a, ScheduledProductHolder& b) {
    a.swap(b);
  }

  class UnscheduledProductHolder : public ProducedProductHolder {
    public:
      explicit UnscheduledProductHolder(std::shared_ptr<BranchDescription const> bd, Principal* principal) :
        ProducedProductHolder(), productData_(bd), theStatus_(UnscheduledNotRun), principal_(principal) {}
      virtual ~UnscheduledProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        UnscheduledProductHolder& other = dynamic_cast<UnscheduledProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void resetStatus_() override {theStatus_ = UnscheduledNotRun;}
      virtual bool onDemand_() const override {return status() == UnscheduledNotRun;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual ProductStatus& status_() const override {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(UnscheduledProductHolder& a, UnscheduledProductHolder& b) {
    a.swap(b);
  }

  class SourceProductHolder : public ProducedProductHolder {
    public:
      explicit SourceProductHolder(std::shared_ptr<BranchDescription const> bd) : ProducedProductHolder(), productData_(bd), theStatus_(NotPut) {}
      virtual ~SourceProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        SourceProductHolder& other = dynamic_cast<SourceProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void resetStatus_() override {theStatus_ = NotPut;}
      virtual bool onDemand_() const override {return false;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual ProductStatus& status_() const override {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
  };

  class AliasProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      explicit AliasProductHolder(std::shared_ptr<BranchDescription const> bd, ProducedProductHolder& realProduct) : ProductHolderBase(), realProduct_(realProduct), bd_(bd) {}
      virtual ~AliasProductHolder();
      ProductStatus& status() const {return realProduct_.status();}
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        AliasProductHolder& other = dynamic_cast<AliasProductHolder&>(rhs);
        realProduct_.swap(other.realProduct_);
        std::swap(bd_, other.bd_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override {return realProduct_.resolveProduct(resolveStatus, skipCurrentProcess, mcc);}
      virtual bool onDemand_() const override {return realProduct_.onDemand();}
      virtual void resetStatus_() override {realProduct_.resetStatus();}
      virtual bool productUnavailable_() const override {return realProduct_.productUnavailable();}
      virtual bool productWasDeleted_() const override {return realProduct_.productWasDeleted();}
      virtual void checkType_(EDProduct const& prod) const override {realProduct_.checkType(prod);}
      virtual ProductData const& getProductData() const override {return realProduct_.productData();}
      virtual ProductData& getProductData() override {return realProduct_.productData();}
      virtual void setProductDeleted_() override {realProduct_.setProductDeleted();}
      virtual void putProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) override {
        realProduct_.putProduct(std::move(edp), productProvenance);
      }
      virtual void putProduct_(std::unique_ptr<EDProduct> edp) const override {
        realProduct_.putProduct(std::move(edp));
      }
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance& productProvenance) override {
        realProduct_.mergeProduct(std::move(edp), productProvenance);
      }
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp) const override {
        realProduct_.mergeProduct(std::move(edp));
      }
      virtual bool putOrMergeProduct_() const override {
        return realProduct_.putOrMergeProduct();
      }
      virtual BranchDescription const& branchDescription_() const override {return *bd_;}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
      virtual std::string const& resolvedModuleLabel_() const override {return realProduct_.moduleLabel();}
      virtual void setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;
      virtual void setPrincipal_(Principal* principal) override;

      ProducedProductHolder& realProduct_;
      std::shared_ptr<BranchDescription const> bd_;
  };

  class NoProcessProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      NoProcessProductHolder(std::vector<ProductHolderIndex> const& matchingHolders,
                             std::vector<bool> const& ambiguous,
                             Principal* principal);
      virtual ~NoProcessProductHolder();
    private:
      virtual ProductData const& getProductData() const override;
      virtual ProductData& getProductData() override;
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void swap_(ProductHolderBase& rhs) override;
      virtual bool onDemand_() const override;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual void putProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance const& productProvenance) override;
      virtual void putProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp, ProductProvenance& productProvenance) override;
      virtual void mergeProduct_(std::unique_ptr<EDProduct> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(EDProduct const& prod) const override;
      virtual void resetStatus_() override;
      virtual void setProductDeleted_() override;
      virtual BranchDescription const& branchDescription_() const override;
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;
      virtual void setPrincipal_(Principal* principal) override;

      std::vector<ProductHolderIndex> matchingHolders_;
      std::vector<bool> ambiguous_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(SourceProductHolder& a, SourceProductHolder& b) {
    a.swap(b);
  }
}

#endif
