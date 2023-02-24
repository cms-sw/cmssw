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
#include "DataFormats/Provenance/interface/BranchDescription.h"
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

    // Give AliasProductResolver and SwitchBaseProductResolver access by moving this method to public
    void resetProductData_(bool deleteEarly) override = 0;
    virtual ProductData const& getProductData() const = 0;
  };

  class DataManagingProductResolver : public DataManagingOrAliasProductResolver {
  public:
    enum class ProductStatus { ProductSet, NotPut, ResolveFailed, ResolveNotRun, ProductDeleted };

    DataManagingProductResolver(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus)
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
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final;

    BranchDescription const& branchDescription_() const final { return *getProductData().branchDescription(); }
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) final {
      productData_.resetBranchDescription(bd);
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
    MergeableInputProductResolver(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus)
        : DataManagingProductResolver(bd, iDefaultStatus) {}

  protected:
    void setOrMergeProduct(std::shared_ptr<WrapperBase> prod,
                           MergeableRunProductMetadata const* mergeableRunProductMetadata) const;

    // merges the product with the pre-existing product
    void mergeProduct(std::shared_ptr<WrapperBase> edp, MergeableRunProductMetadata const*) const;
  };

  class DelayedReaderInputProductResolver : public MergeableInputProductResolver {
  public:
    explicit DelayedReaderInputProductResolver(std::shared_ptr<BranchDescription const> bd)
        : MergeableInputProductResolver(bd, ProductStatus::ResolveNotRun), m_prefetchRequested{false}, aux_{nullptr} {
      assert(bd->onDemand());
      assert(not bd->produced());
    }

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    bool isFromCurrentProcess() const final;

    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;

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
    PutOnReadInputProductResolver(std::shared_ptr<BranchDescription const> bd)
        : MergeableInputProductResolver(bd, ProductStatus::ResolveNotRun) {
      assert(not bd->produced());
      assert(not bd->onDemand());
    }

  protected:
    void putProduct(std::unique_ptr<WrapperBase> edp) const override;
    void putOrMergeProduct(std::unique_ptr<WrapperBase> prod) const override;

    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const final;
    bool unscheduledWasNotRun_() const final { return false; }

  private:
    bool isFromCurrentProcess() const final;
  };

  class ProducedProductResolver : public DataManagingProductResolver, public ProductPutterBase {
  public:
    ProducedProductResolver(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus)
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
    explicit PuttableProductResolver(std::shared_ptr<BranchDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::NotPut) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;
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
    explicit UnscheduledProductResolver(std::shared_ptr<BranchDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::ResolveNotRun) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;
    bool unscheduledWasNotRun_() const override { return status() == ProductStatus::ResolveNotRun; }

    void resetProductData_(bool deleteEarly) override;

    CMS_THREAD_SAFE mutable WaitingTaskList waitingTasks_;
    UnscheduledAuxiliary const* aux_ = nullptr;
    Worker* worker_ = nullptr;
    mutable std::atomic<bool> prefetchRequested_ = false;
  };

  class TransformingProductResolver : public ProducedProductResolver {
  public:
    explicit TransformingProductResolver(std::shared_ptr<BranchDescription const> bd)
        : ProducedProductResolver(bd, ProductStatus::ResolveNotRun), mcc_(nullptr) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    void putProduct(std::unique_ptr<WrapperBase> edp) const override;
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;
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
    explicit AliasProductResolver(std::shared_ptr<BranchDescription const> bd,
                                  DataManagingOrAliasProductResolver& realProduct)
        : DataManagingOrAliasProductResolver(), realProduct_(realProduct), bd_(bd) {}

    void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final {
      realProduct_.connectTo(iOther, iParentPrincipal);
    };

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override {
      return realProduct_.resolveProduct(principal, skipCurrentProcess, sra, mcc);
    }
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override {
      realProduct_.prefetchAsync(waitTask, principal, skipCurrentProcess, token, sra, mcc);
    }
    bool unscheduledWasNotRun_() const override { return realProduct_.unscheduledWasNotRun(); }
    bool productUnavailable_() const override { return realProduct_.productUnavailable(); }
    bool productResolved_() const final { return realProduct_.productResolved(); }
    bool productWasDeleted_() const override { return realProduct_.productWasDeleted(); }
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final {
      return realProduct_.productWasFetchedAndIsValid(iSkipCurrentProcess);
    }

    BranchDescription const& branchDescription_() const override { return *bd_; }
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override { bd_ = bd; }
    Provenance const* provenance_() const final { return realProduct_.provenance(); }

    std::string const& resolvedModuleLabel_() const override { return realProduct_.moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) override;
    void setProductID_(ProductID const& pid) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    ProductData const& getProductData() const final { return realProduct_.getProductData(); }
    bool singleProduct_() const override;

    DataManagingOrAliasProductResolver& realProduct_;
    std::shared_ptr<BranchDescription const> bd_;
  };

  // Switch is a mixture of DataManaging (for worker and provenance) and Alias (for product)
  class SwitchBaseProductResolver : public DataManagingOrAliasProductResolver {
  public:
    using ProductStatus = DataManagingProductResolver::ProductStatus;
    SwitchBaseProductResolver(std::shared_ptr<BranchDescription const> bd,
                              DataManagingOrAliasProductResolver& realProduct);

    void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final;
    void setupUnscheduled(UnscheduledConfigurator const& iConfigure) final;

  protected:
    Resolution resolveProductImpl(Resolution) const;
    WaitingTaskList& waitingTasks() const { return waitingTasks_; }
    Worker* worker() const { return worker_; }
    DataManagingOrAliasProductResolver const& realProduct() const { return realProduct_; }
    std::atomic<bool>& prefetchRequested() const { return prefetchRequested_; }
    void unsafe_setWrapperAndProvenance() const;
    void resetProductData_(bool deleteEarly) override;

  private:
    bool productResolved_() const final;
    bool productWasDeleted_() const final { return realProduct_.productWasDeleted(); }
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final {
      return realProduct_.productWasFetchedAndIsValid(iSkipCurrentProcess);
    }
    BranchDescription const& branchDescription_() const final {
      return *productData_.branchDescription();
      ;
    }
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) final {
      productData_.resetBranchDescription(bd);
    }
    Provenance const* provenance_() const final { return &productData_.provenance(); }
    std::string const& resolvedModuleLabel_() const final { return moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) final;
    void setProductID_(ProductID const& pid) final;
    ProductProvenance const* productProvenancePtr_() const final { return provenance()->productProvenance(); }
    ProductData const& getProductData() const final { return productData_; }
    bool singleProduct_() const final { return true; }

    // for "alias" view
    DataManagingOrAliasProductResolver& realProduct_;
    // for "product" view
    ProductData productData_;
    Worker* worker_ = nullptr;
    CMS_THREAD_SAFE mutable WaitingTaskList waitingTasks_;
    mutable std::atomic<bool> prefetchRequested_;
    // for provenance
    ParentageID parentageID_;
  };

  // For the case when SwitchProducer is on a Path
  class SwitchProducerProductResolver : public SwitchBaseProductResolver, public ProductPutterBase {
  public:
    SwitchProducerProductResolver(std::shared_ptr<BranchDescription const> bd,
                                  DataManagingOrAliasProductResolver& realProduct);

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const final;
    void putProduct(std::unique_ptr<WrapperBase> edp) const final;
    bool unscheduledWasNotRun_() const final { return false; }
    bool productUnavailable_() const final;
    void resetProductData_(bool deleteEarly) final;

    constexpr static const ProductStatus defaultStatus_ = ProductStatus::NotPut;

    // for filter in a Path
    // The variable is only modified or read at times where the
    //  framework has guaranteed synchronization between write and read
    CMS_THREAD_SAFE mutable ProductStatus status_;
  };

  // For the case when SwitchProducer is not on any Path
  class SwitchAliasProductResolver : public SwitchBaseProductResolver {
  public:
    SwitchAliasProductResolver(std::shared_ptr<BranchDescription const> bd,
                               DataManagingOrAliasProductResolver& realProduct)
        : SwitchBaseProductResolver(std::move(bd), realProduct) {}

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const final;
    bool unscheduledWasNotRun_() const final { return realProduct().unscheduledWasNotRun(); }
    bool productUnavailable_() const final { return realProduct().productUnavailable(); }
  };

  class ParentProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    explicit ParentProcessProductResolver(std::shared_ptr<BranchDescription const> bd)
        : ProductResolverBase(), realProduct_(nullptr), bd_(bd), provRetriever_(nullptr), parentPrincipal_(nullptr) {}

    void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final {
      realProduct_ = &iOther;
      parentPrincipal_ = iParentPrincipal;
    };

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override {
      if (principal.branchType() == InProcess &&
          (mcc->parent().globalContext()->transition() == GlobalContext::Transition::kBeginProcessBlock ||
           mcc->parent().globalContext()->transition() == GlobalContext::Transition::kEndProcessBlock)) {
        return Resolution(nullptr);
      }

      skipCurrentProcess = false;
      return realProduct_->resolveProduct(*parentPrincipal_, skipCurrentProcess, sra, mcc);
    }
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override {
      if (principal.branchType() == InProcess &&
          (mcc->parent().globalContext()->transition() == GlobalContext::Transition::kBeginProcessBlock ||
           mcc->parent().globalContext()->transition() == GlobalContext::Transition::kEndProcessBlock)) {
        return;
      }

      skipCurrentProcess = false;
      realProduct_->prefetchAsync(waitTask, *parentPrincipal_, skipCurrentProcess, token, sra, mcc);
    }
    bool unscheduledWasNotRun_() const override {
      if (realProduct_)
        return realProduct_->unscheduledWasNotRun();
      throwNullRealProduct();
      return false;
    }
    bool productUnavailable_() const override { return realProduct_->productUnavailable(); }
    bool productResolved_() const final { return realProduct_->productResolved(); }
    bool productWasDeleted_() const override { return realProduct_->productWasDeleted(); }
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const override {
      iSkipCurrentProcess = false;
      return realProduct_->productWasFetchedAndIsValid(iSkipCurrentProcess);
    }

    BranchDescription const& branchDescription_() const override { return *bd_; }
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override { bd_ = bd; }
    Provenance const* provenance_() const final { return realProduct_->provenance(); }
    std::string const& resolvedModuleLabel_() const override { return realProduct_->moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) override;
    void setProductID_(ProductID const& pid) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    bool singleProduct_() const override;
    void throwNullRealProduct() const;

    ProductResolverBase const* realProduct_;
    std::shared_ptr<BranchDescription const> bd_;
    ProductProvenanceRetriever const* provRetriever_;
    Principal const* parentPrincipal_;
  };

  class NoProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    NoProcessProductResolver(std::vector<ProductResolverIndex> const& matchingHolders,
                             std::vector<bool> const& ambiguous,
                             bool madeAtEnd);

    void connectTo(ProductResolverBase const& iOther, Principal const*) final;

    void tryPrefetchResolverAsync(unsigned int iProcessingIndex,
                                  Principal const& principal,
                                  bool skipCurrentProcess,
                                  SharedResourcesAcquirer* sra,
                                  ModuleCallingContext const* mcc,
                                  ServiceToken token,
                                  oneapi::tbb::task_group*) const;

    bool dataValidFromResolver(unsigned int iProcessingIndex,
                               Principal const& principal,
                               bool iSkipCurrentProcess) const;

    void prefetchFailed(unsigned int iProcessingIndex,
                        Principal const& principal,
                        bool iSkipCurrentProcess,
                        std::exception_ptr iExceptPtr) const;

  private:
    unsigned int unsetIndexValue() const;
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;
    bool unscheduledWasNotRun_() const override;
    bool productUnavailable_() const override;
    bool productWasDeleted_() const override;
    bool productResolved_() const final;
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const override;

    BranchDescription const& branchDescription_() const override;
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
    Provenance const* provenance_() const override;

    std::string const& resolvedModuleLabel_() const override { return moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) override;
    void setProductID_(ProductID const& pid) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    bool singleProduct_() const override;

    Resolution tryResolver(unsigned int index,
                           Principal const& principal,
                           bool skipCurrentProcess,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const;

    void setCache(bool skipCurrentProcess, ProductResolverIndex index, std::exception_ptr exceptionPtr) const;

    std::vector<ProductResolverIndex> matchingHolders_;
    std::vector<bool> ambiguous_;
    CMS_THREAD_SAFE mutable WaitingTaskList waitingTasks_;
    CMS_THREAD_SAFE mutable WaitingTaskList skippingWaitingTasks_;
    mutable std::atomic<unsigned int> lastCheckIndex_;
    mutable std::atomic<unsigned int> lastSkipCurrentCheckIndex_;
    mutable std::atomic<bool> prefetchRequested_;
    mutable std::atomic<bool> skippingPrefetchRequested_;
    const bool madeAtEnd_;
  };

  class SingleChoiceNoProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    SingleChoiceNoProcessProductResolver(ProductResolverIndex iChoice)
        : ProductResolverBase(), realResolverIndex_(iChoice) {}

    void connectTo(ProductResolverBase const& iOther, Principal const*) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const override;
    bool unscheduledWasNotRun_() const override;
    bool productUnavailable_() const override;
    bool productWasDeleted_() const override;
    bool productResolved_() const final;
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const override;

    BranchDescription const& branchDescription_() const override;
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
    Provenance const* provenance_() const override;

    std::string const& resolvedModuleLabel_() const override { return moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) override;
    void setProductID_(ProductID const& pid) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    bool singleProduct_() const override;

    ProductResolverIndex realResolverIndex_;
  };

}  // namespace edm

#endif
