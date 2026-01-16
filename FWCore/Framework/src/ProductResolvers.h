#ifndef FWCore_Framework_ProductResolvers_h
#define FWCore_Framework_ProductResolvers_h

/*----------------------------------------------------------------------

ProductResolver: A collection of information related to a single WrapperBase or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"
#include "FWCore/Framework/src/ProductPutOrMergerBase.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <atomic>
#include <memory>
#include <string>

namespace edm {
  class MergeableRunProductMetadata;
  class ProductProvenanceRetriever;
  class DelayedReader;
  class SharedResourcesAcquirer;
  class UnscheduledAuxiliary;
  class Worker;
  class ServiceToken;

  class DataManagingOrAliasProductResolver : public ProductResolverBase {
  public:
    DataManagingOrAliasProductResolver() : ProductResolverBase{} {}

    // Give AliasProductResolver access by moving this method to public
    void resetProductData_(bool deleteEarly) override = 0;
    virtual ProductData const& getProductData() const = 0;
  };

  class DataManagingProductResolver : public DataManagingOrAliasProductResolver {
  public:
    enum class ProductStatus { ProductSet, NotPut, ResolveFailed, ResolveNotRun, ProductDeleted };

    DataManagingProductResolver(std::shared_ptr<ProductDescription const> bd, ProductStatus iDefaultStatus)
        : DataManagingOrAliasProductResolver(),
          productData_(bd),
          theStatus_(iDefaultStatus),
          defaultStatus_(iDefaultStatus) {}

    void connectTo(ProductResolverBase const&, Principal const*) final;

    void resetStatus() { theStatus_ = defaultStatus_; }

    void resetProductData_(bool deleteEarly) override;

  protected:
    void setProduct(std::unique_ptr<WrapperBase> edp) const;
    void setProduct(std::shared_ptr<WrapperBase> edp) const;
    ProductStatus status() const { return theStatus_; }
    ProductStatus defaultStatus() const { return defaultStatus_; }
    void setFailedStatus() const { theStatus_ = ProductStatus::ResolveFailed; }
    //Handle the boilerplate code needed for resolveProduct_
    template <bool callResolver, typename FUNC>
    Resolution resolveProductImpl(FUNC resolver) const;
    ProductData const& getProductData() const final { return productData_; }
    void setMergeableRunProductMetadataInProductData(MergeableRunProductMetadata const*);

    void checkType(WrapperBase const& prod) const;

  private:
    void throwProductDeletedException() const;
    virtual bool isFromCurrentProcess() const = 0;

    bool productUnavailable_() const final;
    bool productResolved_() const final;
    bool productWasDeleted_() const final;
    bool productWasFetchedAndIsValid_() const final;

    ProductDescription const& productDescription_() const final { return *getProductData().productDescription(); }
    void resetProductDescription_(std::shared_ptr<ProductDescription const> bd) final {
      productData_.resetProductDescription(bd);
    }
    Provenance const* provenance_() const final { return &productData_.provenance(); }

    std::string const& resolvedModuleLabel_() const final { return moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) final;
    void setProductID_(ProductID const& pid) final;
    ProductProvenance const* productProvenancePtr_() const final;
    bool singleProduct_() const final;

    ProductData productData_;
    mutable std::atomic<ProductStatus> theStatus_;
    ProductStatus const defaultStatus_;
  };

  class MergeableInputProductResolver : public DataManagingProductResolver {
  public:
    MergeableInputProductResolver(std::shared_ptr<ProductDescription const> bd, ProductStatus iDefaultStatus)
        : DataManagingProductResolver(bd, iDefaultStatus) {}

  protected:
    void setOrMergeProduct(std::shared_ptr<WrapperBase> prod,
                           MergeableRunProductMetadata const* mergeableRunProductMetadata) const;

    // merges the product with the pre-existing product
    void mergeProduct(std::shared_ptr<WrapperBase> edp, MergeableRunProductMetadata const*) const;
  };

  class DelayedReaderInputProductResolver : public MergeableInputProductResolver {
  public:
    explicit DelayedReaderInputProductResolver(std::shared_ptr<ProductDescription const> bd)
        : MergeableInputProductResolver(bd, ProductStatus::ResolveNotRun), m_prefetchRequested{false}, aux_{nullptr} {
      assert(bd->onDemand());
      assert(not bd->produced());
    }

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    bool isFromCurrentProcess() const final;

    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept override;

    void retrieveAndMerge_(Principal const& principal,
                           MergeableRunProductMetadata const* mergeableRunProductMetadata) const override;

    void setMergeableRunProductMetadata_(MergeableRunProductMetadata const*) override;

    bool unscheduledWasNotRun_() const final { return false; }

    void resetProductData_(bool deleteEarly) override;

    mutable std::atomic<bool> m_prefetchRequested;
    CMS_THREAD_SAFE mutable WaitingTaskList m_waitingTasks;
    UnscheduledAuxiliary const* aux_;  //provides access to the delayedGet signals
  };

  class PutOnReadInputProductResolver : public MergeableInputProductResolver,
                                        public ProductPutterBase,
                                        public ProductPutOrMergerBase {
  public:
    PutOnReadInputProductResolver(std::shared_ptr<ProductDescription const> bd)
        : MergeableInputProductResolver(bd, ProductStatus::ResolveNotRun) {
      assert(not bd->produced());
      assert(not bd->onDemand());
    }

  protected:
    void putProduct(std::unique_ptr<WrapperBase> edp) const override;
    void putOrMergeProduct(std::unique_ptr<WrapperBase> prod) const override;

    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept final;
    bool unscheduledWasNotRun_() const final { return false; }

  private:
    bool isFromCurrentProcess() const final;
  };

  class ProducedProductResolver : public DataManagingProductResolver, public ProductPutterBase {
  public:
    ProducedProductResolver(std::shared_ptr<ProductDescription const> bd, ProductStatus iDefaultStatus)
        : DataManagingProductResolver(bd, iDefaultStatus) {
      assert(bd->produced());
    }

  protected:
    void putProduct(std::unique_ptr<WrapperBase> edp) const override;

  private:
    bool isFromCurrentProcess() const final;
  };

  class PuttableProductResolver : public ProducedProductResolver {
  public:
    explicit PuttableProductResolver(std::shared_ptr<ProductDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::NotPut) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept override;
    bool unscheduledWasNotRun_() const override { return false; }

    // The WaitingTaskList below is the one from the worker, if one
    // corresponds to this ProductResolver. For the Source-like cases
    // where there is no such Worker, the tasks depending on the data
    // depending on this ProductResolver are assumed to be eligible to
    // run immediately after their prefetch.
    WaitingTaskList* waitingTasks_ = nullptr;
  };

  class UnscheduledProductResolver : public ProducedProductResolver {
  public:
    explicit UnscheduledProductResolver(std::shared_ptr<ProductDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::ResolveNotRun) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept override;
    bool unscheduledWasNotRun_() const override { return status() == ProductStatus::ResolveNotRun; }

    void resetProductData_(bool deleteEarly) override;

    CMS_THREAD_SAFE mutable WaitingTaskList waitingTasks_;
    UnscheduledAuxiliary const* aux_ = nullptr;
    Worker* worker_ = nullptr;
    mutable std::atomic<bool> prefetchRequested_ = false;
  };

  class TransformingProductResolver : public ProducedProductResolver {
  public:
    explicit TransformingProductResolver(std::shared_ptr<ProductDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::ResolveNotRun), mcc_(nullptr) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    void putProduct(std::unique_ptr<WrapperBase> edp) const override;
    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept override;
    bool unscheduledWasNotRun_() const override { return status() == ProductStatus::ResolveNotRun; }

    void resetProductData_(bool deleteEarly) override;

    CMS_THREAD_SAFE mutable WaitingTaskList waitingTasks_;
    UnscheduledAuxiliary const* aux_ = nullptr;
    Worker* worker_ = nullptr;
    CMS_THREAD_GUARD(prefetchRequested_) mutable ModuleCallingContext mcc_;
    size_t index_;
    mutable std::atomic<bool> prefetchRequested_ = false;
  };

  class AliasProductResolver : public DataManagingOrAliasProductResolver {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    explicit AliasProductResolver(std::shared_ptr<ProductDescription const> bd,
                                  DataManagingOrAliasProductResolver& realProduct)
        : DataManagingOrAliasProductResolver(), realProduct_(realProduct), bd_(bd) {}

    void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final {
      realProduct_.connectTo(iOther, iParentPrincipal);
    };

  private:
    Resolution resolveProduct_(Principal const& principal,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override {
      return realProduct_.resolveProduct(principal, sra, mcc);
    }
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept override {
      realProduct_.prefetchAsync(waitTask, principal, token, sra, mcc);
    }
    bool unscheduledWasNotRun_() const override { return realProduct_.unscheduledWasNotRun(); }
    bool productUnavailable_() const override { return realProduct_.productUnavailable(); }
    bool productResolved_() const final { return realProduct_.productResolved(); }
    bool productWasDeleted_() const override { return realProduct_.productWasDeleted(); }
    bool productWasFetchedAndIsValid_() const final { return realProduct_.productWasFetchedAndIsValid(); }

    ProductDescription const& productDescription_() const override { return *bd_; }
    void resetProductDescription_(std::shared_ptr<ProductDescription const> bd) override { bd_ = bd; }
    Provenance const* provenance_() const final { return realProduct_.provenance(); }

    std::string const& resolvedModuleLabel_() const override { return realProduct_.moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) override;
    void setProductID_(ProductID const& pid) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    ProductData const& getProductData() const final { return realProduct_.getProductData(); }
    bool singleProduct_() const override;

    DataManagingOrAliasProductResolver& realProduct_;
    std::shared_ptr<ProductDescription const> bd_;
  };
}  // namespace edm

#endif
