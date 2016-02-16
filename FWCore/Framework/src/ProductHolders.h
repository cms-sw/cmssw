#ifndef FWCore_Framework_ProductHolders_h
#define FWCore_Framework_ProductHolders_h

/*----------------------------------------------------------------------

ProductHolder: A collection of information related to a single WrapperBase or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductHolderBase.h"
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

  class InputProductHolder : public ProductHolderBase {
    public:
    explicit InputProductHolder(std::shared_ptr<BranchDescription const> bd) :
        ProductHolderBase(), productData_(bd),
        productHasBeenDeleted_(false) {}

    private:
      // The following is const because we can add an EDProduct to the
      // cache after creation of the ProductHolder, without changing the meaning
      // of the ProductHolder.
      void setProduct(std::unique_ptr<WrapperBase> prod) const;
    
      virtual void swap_(ProductHolderBase& rhs) override {
        InputProductHolder& other = dynamic_cast<InputProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp, ProductProvenance const& productProvenance) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(WrapperBase const&) const override {}
      virtual void resetStatus_() override {productHasBeenDeleted_=false;}
      virtual bool onDemand_() const override {return false;}
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override {return productHasBeenDeleted_;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual void setProductDeleted_() const override {productHasBeenDeleted_ = true;}
      virtual BranchDescription const& branchDescription_() const override {return *productData_.branchDescription();}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {productData_.resetBranchDescription(bd);}
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;

      ProductData productData_;
      mutable bool productHasBeenDeleted_;
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
      void producerStarted();
      void producerCompleted();
      ProductStatus& status() const {return status_();}


    private:
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp, ProductProvenance const& productProvenance) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(WrapperBase const& prod) const override {
        reallyCheckType(prod);
      }
      virtual ProductStatus& status_() const = 0;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual void setProductDeleted_() const override;
      virtual BranchDescription const& branchDescription_() const override {return *getProductData().branchDescription();}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {getProductData().resetBranchDescription(bd);}
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;
  };

  class PuttableProductHolder : public ProducedProductHolder {
  public:
    explicit PuttableProductHolder(std::shared_ptr<BranchDescription const> bd, ProductStatus defaultStatus) : ProducedProductHolder(), productData_(bd), theStatus_(defaultStatus), defaultStatus_(defaultStatus) {}

  private:
    virtual void swap_(ProductHolderBase& rhs) override {
      auto& other = dynamic_cast<PuttableProductHolder&>(rhs);
      edm::swap(productData_, other.productData_);
      std::swap(theStatus_, other.theStatus_);
    }
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const override;
    virtual void resetStatus_() override {theStatus_ = defaultStatus_;}
    virtual bool onDemand_() const override {return false;}
    virtual ProductData const& getProductData() const override {return productData_;}
    virtual ProductData& getProductData() override {return productData_;}
    virtual ProductStatus& status_() const override {return theStatus_;}
    
    ProductData productData_;
    mutable ProductStatus theStatus_;
    const ProductStatus defaultStatus_;
  };
  
  class UnscheduledProductHolder : public ProducedProductHolder {
    public:
      explicit UnscheduledProductHolder(std::shared_ptr<BranchDescription const> bd) :
        ProducedProductHolder(), productData_(bd), theStatus_(UnscheduledNotRun) {}
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        UnscheduledProductHolder& other = dynamic_cast<UnscheduledProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void resetStatus_() override {theStatus_ = UnscheduledNotRun;}
      virtual bool onDemand_() const override {return status() == UnscheduledNotRun;}
      virtual ProductData const& getProductData() const override {return productData_;}
      virtual ProductData& getProductData() override {return productData_;}
      virtual ProductStatus& status_() const override {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
  };

  // Free swap function
  inline void swap(UnscheduledProductHolder& a, UnscheduledProductHolder& b) {
    a.swap(b);
  }

  class AliasProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      explicit AliasProductHolder(std::shared_ptr<BranchDescription const> bd, ProducedProductHolder& realProduct) : ProductHolderBase(), realProduct_(realProduct), bd_(bd) {}
      ProductStatus& status() const {return realProduct_.status();}
    private:
      virtual void swap_(ProductHolderBase& rhs) override {
        AliasProductHolder& other = dynamic_cast<AliasProductHolder&>(rhs);
        realProduct_.swap(other.realProduct_);
        std::swap(bd_, other.bd_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override {return realProduct_.resolveProduct(resolveStatus, principal, skipCurrentProcess, sra, mcc);}
      virtual bool onDemand_() const override {return realProduct_.onDemand();}
      virtual void resetStatus_() override {realProduct_.resetStatus();}
      virtual bool productUnavailable_() const override {return realProduct_.productUnavailable();}
      virtual bool productWasDeleted_() const override {return realProduct_.productWasDeleted();}
      virtual void checkType_(WrapperBase const& prod) const override {realProduct_.checkType(prod);}
      virtual ProductData const& getProductData() const override {return realProduct_.getProductData();}
      virtual ProductData& getProductData() override {return realProduct_.getProductData();}
      virtual void setProductDeleted_() const override {realProduct_.setProductDeleted();}
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp, ProductProvenance const& productProvenance) const override {
        realProduct_.putProduct(std::move(edp), productProvenance);
      }
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override {
        realProduct_.putProduct(std::move(edp));
      }
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override {
        realProduct_.mergeProduct(std::move(edp));
      }
      virtual bool putOrMergeProduct_() const override {
        return realProduct_.putOrMergeProduct();
      }
      virtual BranchDescription const& branchDescription_() const override {return *bd_;}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
      virtual std::string const& resolvedModuleLabel_() const override {return realProduct_.moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;

      ProducedProductHolder& realProduct_;
      std::shared_ptr<BranchDescription const> bd_;
  };

  class NoProcessProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      NoProcessProductHolder(std::vector<ProductHolderIndex> const& matchingHolders,
                             std::vector<bool> const& ambiguous);
    private:
      virtual ProductData const& getProductData() const override;
      virtual ProductData& getProductData() override;
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void swap_(ProductHolderBase& rhs) override;
      virtual bool onDemand_() const override;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp, ProductProvenance const& productProvenance) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(WrapperBase const& prod) const override;
      virtual void resetStatus_() override;
      virtual void setProductDeleted_() const override;
      virtual BranchDescription const& branchDescription_() const override;
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_() override;
      virtual bool singleProduct_() const override;

      std::vector<ProductHolderIndex> matchingHolders_;
      std::vector<bool> ambiguous_;
  };

}

#endif
