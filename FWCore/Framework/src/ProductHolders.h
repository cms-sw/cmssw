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
#include <atomic>

#include <string>

namespace edm {
  class ProductProvenanceRetriever;
  class DelayedReader;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class Principal;
  
  class DataManagingProductHolder : public ProductHolderBase {
  public:
    enum class ProductStatus {
      ProductSet,
      NotPut,
      ResolveFailed,
      ResolveNotRun,
      ProductDeleted
    };
    
    DataManagingProductHolder(std::shared_ptr<BranchDescription const> bd,ProductStatus iDefaultStatus): ProductHolderBase(),
    productData_(bd),
    theStatus_(iDefaultStatus),
    defaultStatus_(iDefaultStatus){}
    
    virtual void connectTo(ProductHolderBase const&, Principal const*) override final;

    void resetStatus() {theStatus_ = defaultStatus_;}
    
    //Give AliasProductHolder access
    virtual void resetProductData_(bool deleteEarly) override final;

  protected:
    ProductData const& getProductData() const {return productData_;}
    void setProduct(std::unique_ptr<WrapperBase> edp) const;
    ProductStatus status() const { return theStatus_;}
    ProductStatus defaultStatus() const { return defaultStatus_; }
    void setFailedStatus() const { theStatus_ = ProductStatus::ResolveFailed; }
    
    
  private:
    virtual void swap_(ProductHolderBase& rhs) override {
      auto& other = dynamic_cast<DataManagingProductHolder&>(rhs);
      edm::swap(productData_, other.productData_);
      theStatus_.store(other.theStatus_.exchange(theStatus_.load()));
    }

    virtual void checkType_(WrapperBase const& prod) const override {
      reallyCheckType(prod);
    }
    virtual bool productUnavailable_() const override ;
    virtual bool productResolved_() const override final;
    virtual bool productWasDeleted_() const override final;
    virtual BranchDescription const& branchDescription_() const override final {return *getProductData().branchDescription();}
    virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override final {productData_.resetBranchDescription(bd);}
    virtual Provenance const* provenance_() const override final {return &productData_.provenance();}

    virtual std::string const& resolvedModuleLabel_() const override final {return moduleLabel();}
    virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override final;
    virtual void setProcessHistory_(ProcessHistory const& ph) override final;
    virtual ProductProvenance const* productProvenancePtr_() const override final;
    virtual bool singleProduct_() const override final;

    ProductData productData_;
    mutable std::atomic<ProductStatus> theStatus_;
    ProductStatus const defaultStatus_;
  };

  class InputProductHolder : public DataManagingProductHolder {
    public:
    explicit InputProductHolder(std::shared_ptr<BranchDescription const> bd) :
      DataManagingProductHolder(bd, ProductStatus::ResolveNotRun) {}

    private:
    
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual bool onDemandWasNotRun_() const override final {return false;}
      virtual bool productUnavailable_() const override;

  };

  // Free swap function
  inline void swap(InputProductHolder& a, InputProductHolder& b) {
    a.swap(b);
  }

  class ProducedProductHolder : public DataManagingProductHolder {
    public:
      ProducedProductHolder(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus) : DataManagingProductHolder(bd, iDefaultStatus) {}

    private:
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
  };

  class PuttableProductHolder : public ProducedProductHolder {
  public:
    explicit PuttableProductHolder(std::shared_ptr<BranchDescription const> bd) : ProducedProductHolder(bd, ProductStatus::NotPut) {}

  private:
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const override;
    virtual bool onDemandWasNotRun_() const override {return false;}
  };
  
  class UnscheduledProductHolder : public ProducedProductHolder {
    public:
      explicit UnscheduledProductHolder(std::shared_ptr<BranchDescription const> bd) :
       ProducedProductHolder(bd,ProductStatus::ResolveNotRun) {}
    private:
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual bool onDemandWasNotRun_() const override {return status() == ProductStatus::ResolveNotRun;}
  };

  // Free swap function
  inline void swap(UnscheduledProductHolder& a, UnscheduledProductHolder& b) {
    a.swap(b);
  }

  class AliasProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      explicit AliasProductHolder(std::shared_ptr<BranchDescription const> bd, ProducedProductHolder& realProduct) : ProductHolderBase(), realProduct_(realProduct), bd_(bd) {}
    
      virtual void connectTo(ProductHolderBase const& iOther, Principal const* iParentPrincipal) override final {
        realProduct_.connectTo(iOther, iParentPrincipal );
      };

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
      virtual bool onDemandWasNotRun_() const override {return realProduct_.onDemandWasNotRun();}
      virtual bool productUnavailable_() const override {return realProduct_.productUnavailable();}
      virtual bool productResolved_() const override final {
          return realProduct_.productResolved(); }
      virtual bool productWasDeleted_() const override {return realProduct_.productWasDeleted();}
      virtual void checkType_(WrapperBase const& prod) const override {realProduct_.checkType(prod);}
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
      virtual Provenance const* provenance_() const override final { return realProduct_.provenance(); }

      virtual std::string const& resolvedModuleLabel_() const override {return realProduct_.moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_(bool deleteEarly) override;
      virtual bool singleProduct_() const override;

      ProducedProductHolder& realProduct_;
      std::shared_ptr<BranchDescription const> bd_;
  };

  class ParentProcessProductHolder : public ProductHolderBase {
  public:
    typedef ProducedProductHolder::ProductStatus ProductStatus;
    explicit ParentProcessProductHolder(std::shared_ptr<BranchDescription const> bd) : ProductHolderBase(), realProduct_(nullptr), bd_(bd), provRetriever_(nullptr), parentPrincipal_(nullptr) {}
    
    virtual void connectTo(ProductHolderBase const& iOther, Principal const* iParentPrincipal) override final {
      realProduct_ = &iOther;
      parentPrincipal_ = iParentPrincipal;
    };
    
  private:
    virtual void swap_(ProductHolderBase& rhs) override {
      ParentProcessProductHolder& other = dynamic_cast<ParentProcessProductHolder&>(rhs);
      std::swap(realProduct_,other.realProduct_);
      std::swap(bd_, other.bd_);
    }
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const override {return realProduct_->resolveProduct(resolveStatus, *parentPrincipal_, skipCurrentProcess, sra, mcc);}
    virtual bool onDemandWasNotRun_() const override {return realProduct_->onDemandWasNotRun();}
    virtual bool productUnavailable_() const override {return realProduct_->productUnavailable();}
    virtual bool productResolved_() const override final { return realProduct_->productResolved(); }
    virtual bool productWasDeleted_() const override {return realProduct_->productWasDeleted();}
    virtual void checkType_(WrapperBase const& prod) const override {realProduct_->checkType(prod);}
    virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override {
      realProduct_->putProduct(std::move(edp));
    }
    virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override {
      realProduct_->mergeProduct(std::move(edp));
    }
    virtual bool putOrMergeProduct_() const override {
      return realProduct_->putOrMergeProduct();
    }
    virtual BranchDescription const& branchDescription_() const override {return *bd_;}
    virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
    virtual Provenance const* provenance_() const override final {return realProduct_->provenance();
    }
    virtual std::string const& resolvedModuleLabel_() const override {return realProduct_->moduleLabel();}
    virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
    virtual void setProcessHistory_(ProcessHistory const& ph) override;
    virtual ProductProvenance const* productProvenancePtr_() const override;
    virtual void resetProductData_(bool deleteEarly) override;
    virtual bool singleProduct_() const override;
    
    ProductHolderBase const* realProduct_;
    std::shared_ptr<BranchDescription const> bd_;
    ProductProvenanceRetriever const* provRetriever_;
    Principal const* parentPrincipal_;
  };

  class NoProcessProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      NoProcessProductHolder(std::vector<ProductHolderIndex> const& matchingHolders,
                             std::vector<bool> const& ambiguous);
    
    virtual void connectTo(ProductHolderBase const& iOther, Principal const*) override final ;

    private:
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void swap_(ProductHolderBase& rhs) override;
      virtual bool onDemandWasNotRun_() const override;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual bool productResolved_() const override final;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void mergeProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool putOrMergeProduct_() const override;
      virtual void checkType_(WrapperBase const& prod) const override;
      virtual BranchDescription const& branchDescription_() const override;
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
      virtual Provenance const* provenance_() const override;

      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_(bool deleteEarly) override;
      virtual bool singleProduct_() const override;

      std::vector<ProductHolderIndex> matchingHolders_;
      std::vector<bool> ambiguous_;
  };

}

#endif
