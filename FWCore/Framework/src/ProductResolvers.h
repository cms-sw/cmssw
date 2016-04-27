#ifndef FWCore_Framework_ProductResolvers_h
#define FWCore_Framework_ProductResolvers_h

/*----------------------------------------------------------------------

ProductResolver: A collection of information related to a single WrapperBase or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
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
  class UnscheduledAuxiliary;
  class Worker;
  
  class DataManagingProductResolver : public ProductResolverBase {
  public:
    enum class ProductStatus {
      ProductSet,
      NotPut,
      ResolveFailed,
      ResolveNotRun,
      ProductDeleted
    };
    
    DataManagingProductResolver(std::shared_ptr<BranchDescription const> bd,ProductStatus iDefaultStatus): ProductResolverBase(),
    productData_(bd),
    theStatus_(iDefaultStatus),
    defaultStatus_(iDefaultStatus){}
    
    virtual void connectTo(ProductResolverBase const&, Principal const*) override final;

    void resetStatus() {theStatus_ = defaultStatus_;}
    
    //Give AliasProductResolver access
    virtual void resetProductData_(bool deleteEarly) override final;

  protected:
    void setProduct(std::unique_ptr<WrapperBase> edp) const;
    ProductStatus status() const { return theStatus_;}
    ProductStatus defaultStatus() const { return defaultStatus_; }
    void setFailedStatus() const { theStatus_ = ProductStatus::ResolveFailed; }
    //Handle the boilerplate code needed for resolveProduct_
    template <bool callResolver, typename FUNC>
    ProductData const* resolveProductImpl( FUNC resolver, ResolveStatus& resolveStatus) const;
    
  private:

    void throwProductDeletedException() const;
    void checkType(WrapperBase const& prod) const;
    ProductData const& getProductData() const {return productData_;}
    // merges the product with the pre-existing product
    void mergeProduct(std::unique_ptr<WrapperBase> edp) const;

    virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const override final;
    virtual bool productUnavailable_() const override final;
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

  class InputProductResolver : public DataManagingProductResolver {
    public:
    explicit InputProductResolver(std::shared_ptr<BranchDescription const> bd) :
      DataManagingProductResolver(bd, ProductStatus::ResolveNotRun) {}

    private:
    
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual bool unscheduledWasNotRun_() const override final {return false;}

  };

  class ProducedProductResolver : public DataManagingProductResolver {
    public:
      ProducedProductResolver(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus) : DataManagingProductResolver(bd, iDefaultStatus) {assert(bd->produced());}

    private:
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
  };

  class PuttableProductResolver : public ProducedProductResolver {
  public:
    explicit PuttableProductResolver(std::shared_ptr<BranchDescription const> bd) : ProducedProductResolver(bd, ProductStatus::NotPut) {}

  private:
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const override;
    virtual bool unscheduledWasNotRun_() const override {return false;}
  };
  
  class UnscheduledProductResolver : public ProducedProductResolver {
    public:
      explicit UnscheduledProductResolver(std::shared_ptr<BranchDescription const> bd) :
       ProducedProductResolver(bd,ProductStatus::ResolveNotRun),
       aux_(nullptr){}
    
      virtual void setupUnscheduled(UnscheduledConfigurator const&) override final;

    private:
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual bool unscheduledWasNotRun_() const override {return status() == ProductStatus::ResolveNotRun;}
    
      UnscheduledAuxiliary const* aux_;
      Worker* worker_;
  };

  class AliasProductResolver : public ProductResolverBase {
    public:
      typedef ProducedProductResolver::ProductStatus ProductStatus;
      explicit AliasProductResolver(std::shared_ptr<BranchDescription const> bd, ProducedProductResolver& realProduct) : ProductResolverBase(), realProduct_(realProduct), bd_(bd) {}
    
      virtual void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) override final {
        realProduct_.connectTo(iOther, iParentPrincipal );
      };

    private:
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override {return realProduct_.resolveProduct(resolveStatus, principal, skipCurrentProcess, sra, mcc);}
      virtual bool unscheduledWasNotRun_() const override {return realProduct_.unscheduledWasNotRun();}
      virtual bool productUnavailable_() const override {return realProduct_.productUnavailable();}
      virtual bool productResolved_() const override final {
          return realProduct_.productResolved(); }
      virtual bool productWasDeleted_() const override {return realProduct_.productWasDeleted();}
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const override final;
      virtual BranchDescription const& branchDescription_() const override {return *bd_;}
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
      virtual Provenance const* provenance_() const override final { return realProduct_.provenance(); }

      virtual std::string const& resolvedModuleLabel_() const override {return realProduct_.moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_(bool deleteEarly) override;
      virtual bool singleProduct_() const override;

      ProducedProductResolver& realProduct_;
      std::shared_ptr<BranchDescription const> bd_;
  };

  class ParentProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    explicit ParentProcessProductResolver(std::shared_ptr<BranchDescription const> bd) : ProductResolverBase(), realProduct_(nullptr), bd_(bd), provRetriever_(nullptr), parentPrincipal_(nullptr) {}
    
    virtual void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) override final {
      realProduct_ = &iOther;
      parentPrincipal_ = iParentPrincipal;
    };
    
  private:
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                               Principal const& principal,
                                               bool skipCurrentProcess,
                                               SharedResourcesAcquirer* sra,
                                               ModuleCallingContext const* mcc) const override {return realProduct_->resolveProduct(resolveStatus, *parentPrincipal_, skipCurrentProcess, sra, mcc);}
    virtual bool unscheduledWasNotRun_() const override {return realProduct_->unscheduledWasNotRun();}
    virtual bool productUnavailable_() const override {return realProduct_->productUnavailable();}
    virtual bool productResolved_() const override final { return realProduct_->productResolved(); }
    virtual bool productWasDeleted_() const override {return realProduct_->productWasDeleted();}
    virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
    virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const override final;
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
    
    ProductResolverBase const* realProduct_;
    std::shared_ptr<BranchDescription const> bd_;
    ProductProvenanceRetriever const* provRetriever_;
    Principal const* parentPrincipal_;
  };

  class NoProcessProductResolver : public ProductResolverBase {
    public:
      typedef ProducedProductResolver::ProductStatus ProductStatus;
      NoProcessProductResolver(std::vector<ProductResolverIndex> const& matchingHolders,
                             std::vector<bool> const& ambiguous);
    
    virtual void connectTo(ProductResolverBase const& iOther, Principal const*) override final ;

    private:
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus,
                                                 Principal const& principal,
                                                 bool skipCurrentProcess,
                                                 SharedResourcesAcquirer* sra,
                                                 ModuleCallingContext const* mcc) const override;
      virtual bool unscheduledWasNotRun_() const override;
      virtual bool productUnavailable_() const override;
      virtual bool productWasDeleted_() const override;
      virtual bool productResolved_() const override final;
      virtual void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      virtual void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const override final;
      virtual BranchDescription const& branchDescription_() const override;
      virtual void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
      virtual Provenance const* provenance_() const override;

      virtual std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      virtual void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      virtual void setProcessHistory_(ProcessHistory const& ph) override;
      virtual ProductProvenance const* productProvenancePtr_() const override;
      virtual void resetProductData_(bool deleteEarly) override;
      virtual bool singleProduct_() const override;

      std::vector<ProductResolverIndex> matchingHolders_;
      std::vector<bool> ambiguous_;
  };

}

#endif
