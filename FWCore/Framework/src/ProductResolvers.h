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
#include "FWCore/Concurrency/interface/WaitingTaskList.h"

#include <memory>
#include <atomic>

#include <string>

namespace edm {
  class MergeableRunProductMetadata;
  class ProductProvenanceRetriever;
  class DelayedReader;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class Principal;
  class UnscheduledAuxiliary;
  class Worker;
  class ServiceToken;

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

    void connectTo(ProductResolverBase const&, Principal const*) final;

    void resetStatus() {theStatus_ = defaultStatus_;}

    //Give AliasProductResolver access
    void resetProductData_(bool deleteEarly) override;

  protected:
    void setProduct(std::unique_ptr<WrapperBase> edp) const;
    ProductStatus status() const { return theStatus_;}
    ProductStatus defaultStatus() const { return defaultStatus_; }
    void setFailedStatus() const { theStatus_ = ProductStatus::ResolveFailed; }
    //Handle the boilerplate code needed for resolveProduct_
    template <bool callResolver, typename FUNC>
    Resolution resolveProductImpl( FUNC resolver) const;
    void setMergeableRunProductMetadataInProductData(MergeableRunProductMetadata const*);

  private:

    void throwProductDeletedException() const;
    void checkType(WrapperBase const& prod) const;
    ProductData const& getProductData() const {return productData_;}
    virtual bool isFromCurrentProcess() const = 0;
    // merges the product with the pre-existing product
    void mergeProduct(std::unique_ptr<WrapperBase> edp, MergeableRunProductMetadata const*) const;

    void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
    bool productUnavailable_() const final;
    bool productResolved_() const final;
    bool productWasDeleted_() const final;
    bool productWasFetchedAndIsValid_( bool iSkipCurrentProcess) const final;

    BranchDescription const& branchDescription_() const final {return *getProductData().branchDescription();}
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) final {productData_.resetBranchDescription(bd);}
    Provenance const* provenance_() const final {return &productData_.provenance();}

    std::string const& resolvedModuleLabel_() const final {return moduleLabel();}
    void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) final;
    void setProcessHistory_(ProcessHistory const& ph) final;
    ProductProvenance const* productProvenancePtr_() const final;
    bool singleProduct_() const final;

    ProductData productData_;
    mutable std::atomic<ProductStatus> theStatus_;
    ProductStatus const defaultStatus_;
  };

  class InputProductResolver : public DataManagingProductResolver {
    public:
    explicit InputProductResolver(std::shared_ptr<BranchDescription const> bd) :
      DataManagingProductResolver(bd, ProductStatus::ResolveNotRun),
      m_prefetchRequested{ false },
      aux_{nullptr} {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

    private:
      bool isFromCurrentProcess() const final;


      Resolution resolveProduct_(Principal const& principal,
                                 bool skipCurrentProcess,
                                 SharedResourcesAcquirer* sra,
                                 ModuleCallingContext const* mcc) const override;
     void prefetchAsync_(WaitingTask* waitTask,
                         Principal const& principal,
                         bool skipCurrentProcess,
                         ServiceToken const& token,
                         SharedResourcesAcquirer* sra,
                         ModuleCallingContext const* mcc) const override;
      void putProduct_(std::unique_ptr<WrapperBase> edp) const override;

      void retrieveAndMerge_(Principal const& principal, MergeableRunProductMetadata const* mergeableRunProductMetadata) const override;

      void setMergeableRunProductMetadata_(MergeableRunProductMetadata const*) override;

      bool unscheduledWasNotRun_() const final {return false;}

      void resetProductData_(bool deleteEarly) override;

      mutable std::atomic<bool> m_prefetchRequested;
      mutable WaitingTaskList m_waitingTasks;
      UnscheduledAuxiliary const* aux_; //provides access to the delayedGet signals


  };

  class ProducedProductResolver : public DataManagingProductResolver {
    public:
      ProducedProductResolver(std::shared_ptr<BranchDescription const> bd, ProductStatus iDefaultStatus) : DataManagingProductResolver(bd, iDefaultStatus) {assert(bd->produced());}

    protected:
      void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
    private:
      bool isFromCurrentProcess() const final;

  };

  class PuttableProductResolver : public ProducedProductResolver {
  public:
    explicit PuttableProductResolver(std::shared_ptr<BranchDescription const> bd) : ProducedProductResolver(bd, ProductStatus::NotPut), worker_(nullptr), prefetchRequested_(false) {}

    void setupUnscheduled(UnscheduledConfigurator const&) final;

  private:
    Resolution resolveProduct_(Principal const& principal,
                                       bool skipCurrentProcess,
                                       SharedResourcesAcquirer* sra,
                                       ModuleCallingContext const* mcc) const override;
     void prefetchAsync_(WaitingTask* waitTask,
                                 Principal const& principal,
                                 bool skipCurrentProcess,
                         ServiceToken const& token,
                                 SharedResourcesAcquirer* sra,
                                 ModuleCallingContext const* mcc) const override;
    bool unscheduledWasNotRun_() const override {return false;}

    void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
    void resetProductData_(bool deleteEarly) override;

    mutable WaitingTaskList m_waitingTasks;
    Worker* worker_;
    mutable std::atomic<bool> prefetchRequested_;

  };

  class UnscheduledProductResolver : public ProducedProductResolver {
    public:
      explicit UnscheduledProductResolver(std::shared_ptr<BranchDescription const> bd) :
       ProducedProductResolver(bd,ProductStatus::ResolveNotRun),
       aux_(nullptr),
       prefetchRequested_(false){}

      void setupUnscheduled(UnscheduledConfigurator const&) final;

    private:
      Resolution resolveProduct_(Principal const& principal,
                                         bool skipCurrentProcess,
                                         SharedResourcesAcquirer* sra,
                                         ModuleCallingContext const* mcc) const override;
       void prefetchAsync_(WaitingTask* waitTask,
                           Principal const& principal,
                           bool skipCurrentProcess,
                           ServiceToken const& token,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const override;
      bool unscheduledWasNotRun_() const override {return status() == ProductStatus::ResolveNotRun;}

      void resetProductData_(bool deleteEarly) override;

      mutable WaitingTaskList waitingTasks_;
      UnscheduledAuxiliary const* aux_;
      Worker* worker_;
      mutable std::atomic<bool> prefetchRequested_;
  };

  class AliasProductResolver : public ProductResolverBase {
    public:
      typedef ProducedProductResolver::ProductStatus ProductStatus;
      explicit AliasProductResolver(std::shared_ptr<BranchDescription const> bd, ProducedProductResolver& realProduct) : ProductResolverBase(), realProduct_(realProduct), bd_(bd) {}

      void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final {
        realProduct_.connectTo(iOther, iParentPrincipal );
      };

    private:
      Resolution resolveProduct_(Principal const& principal,
                                 bool skipCurrentProcess,
                                 SharedResourcesAcquirer* sra,
                                 ModuleCallingContext const* mcc) const override {
        return realProduct_.resolveProduct(principal, skipCurrentProcess, sra, mcc);}
       void prefetchAsync_(WaitingTask* waitTask,
                           Principal const& principal,
                           bool skipCurrentProcess,
                           ServiceToken const& token,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const override {
        realProduct_.prefetchAsync(waitTask, principal, skipCurrentProcess, token, sra, mcc);
      }
      bool unscheduledWasNotRun_() const override {return realProduct_.unscheduledWasNotRun();}
      bool productUnavailable_() const override {return realProduct_.productUnavailable();}
      bool productResolved_() const final {
          return realProduct_.productResolved(); }
      bool productWasDeleted_() const override {return realProduct_.productWasDeleted();}
      bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final {
        return realProduct_.productWasFetchedAndIsValid(iSkipCurrentProcess);
      }

      void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
      BranchDescription const& branchDescription_() const override {return *bd_;}
      void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
      Provenance const* provenance_() const final { return realProduct_.provenance(); }

      std::string const& resolvedModuleLabel_() const override {return realProduct_.moduleLabel();}
      void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      void setProcessHistory_(ProcessHistory const& ph) override;
      ProductProvenance const* productProvenancePtr_() const override;
      void resetProductData_(bool deleteEarly) override;
      bool singleProduct_() const override;

      ProducedProductResolver& realProduct_;
      std::shared_ptr<BranchDescription const> bd_;
  };

  // Switch is a mixture of DataManaging (for worker and provenance) and Alias (for product)
  class SwitchBaseProductResolver: public ProductResolverBase {
  public:
    using ProductStatus = DataManagingProductResolver::ProductStatus;
    SwitchBaseProductResolver(std::shared_ptr<BranchDescription const> bd, ProducedProductResolver& realProduct);

    void connectTo(ProductResolverBase const& iOther, Principal const *iParentPrincipal) final;
    void setupUnscheduled(UnscheduledConfigurator const& iConfigure) final;

  protected:
    Resolution resolveProductImpl(Resolution) const;
    WaitingTaskList& waitingTasks() const {return waitingTasks_;}
    Worker *worker() const {return worker_;}
    ProductStatus status() const {return status_;}
    ProducedProductResolver const& realProduct() const {return realProduct_;}
    std::atomic<bool>& prefetchRequested() const {return prefetchRequested_;}
    
  private:
    bool productResolved_() const final;
    bool productWasDeleted_() const final { return realProduct_.productWasDeleted(); }
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final { return realProduct_.productWasFetchedAndIsValid(iSkipCurrentProcess); }
    void putProduct_(std::unique_ptr<WrapperBase> edp) const final;
    void putOrMergeProduct_(std::unique_ptr<WrapperBase> edp, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
    BranchDescription const& branchDescription_() const final {return *productData_.branchDescription();;}
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) final {productData_.resetBranchDescription(bd);}
    Provenance const* provenance_() const final {return &productData_.provenance();}
    std::string const& resolvedModuleLabel_() const final {return moduleLabel();}
    void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) final;
    void setProcessHistory_(ProcessHistory const& ph) final {productData_.setProcessHistory(ph);}
    ProductProvenance const* productProvenancePtr_() const final {return provenance()->productProvenance();}
    void resetProductData_(bool deleteEarly) final;
    bool singleProduct_() const final {return true;}

    constexpr static const ProductStatus defaultStatus_ = ProductStatus::NotPut;

    // for "alias" view
    ProducedProductResolver& realProduct_;
    // for "product" view
    ProductData productData_;
    Worker *worker_ = nullptr;
    mutable WaitingTaskList waitingTasks_;
    mutable std::atomic<bool> prefetchRequested_;
    // for provenance
    ParentageID parentageID_;
    // for filter in a Path
    mutable ProductStatus status_;
  };

  // For the case when SwitchProducer is on a Path
  class SwitchProducerProductResolver: public SwitchBaseProductResolver {
  public:
    SwitchProducerProductResolver(std::shared_ptr<BranchDescription const> bd, ProducedProductResolver& realProduct):
      SwitchBaseProductResolver(std::move(bd), realProduct) {}
  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTask* waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const final;
    bool unscheduledWasNotRun_() const final { return false; }
    bool productUnavailable_() const final;
  };

  // For the case when SwitchProducer is not on any Path
  class SwitchAliasProductResolver: public SwitchBaseProductResolver {
  public:
    SwitchAliasProductResolver(std::shared_ptr<BranchDescription const> bd, ProducedProductResolver& realProduct):
      SwitchBaseProductResolver(std::move(bd), realProduct) {}
  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTask* waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const final;
    bool unscheduledWasNotRun_() const final {return realProduct().unscheduledWasNotRun();}
    bool productUnavailable_() const final {return realProduct().productUnavailable();}
  };

  class ParentProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    explicit ParentProcessProductResolver(std::shared_ptr<BranchDescription const> bd) : ProductResolverBase(), realProduct_(nullptr), bd_(bd), provRetriever_(nullptr), parentPrincipal_(nullptr) {}

    void connectTo(ProductResolverBase const& iOther, Principal const* iParentPrincipal) final {
      realProduct_ = &iOther;
      parentPrincipal_ = iParentPrincipal;
    };

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override {
      skipCurrentProcess = false;
      return realProduct_->resolveProduct(*parentPrincipal_, skipCurrentProcess, sra, mcc);
    }
     void prefetchAsync_(WaitingTask* waitTask,
                         Principal const& principal,
                         bool skipCurrentProcess,
                         ServiceToken const& token,
                         SharedResourcesAcquirer* sra,
                         ModuleCallingContext const* mcc) const override {
      skipCurrentProcess = false;
      realProduct_->prefetchAsync( waitTask, *parentPrincipal_, skipCurrentProcess, token, sra, mcc);
    }
    bool unscheduledWasNotRun_() const override {
      if (realProduct_) return realProduct_->unscheduledWasNotRun();
      throwNullRealProduct();
      return false;
    }
    bool productUnavailable_() const override {return realProduct_->productUnavailable();}
    bool productResolved_() const final { return realProduct_->productResolved(); }
    bool productWasDeleted_() const override {return realProduct_->productWasDeleted();}
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const override {
      iSkipCurrentProcess = false;
      return realProduct_->productWasFetchedAndIsValid(iSkipCurrentProcess);
    }

    void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
    void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
    BranchDescription const& branchDescription_() const override {return *bd_;}
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override {bd_ = bd;}
    Provenance const* provenance_() const final {return realProduct_->provenance();
    }
    std::string const& resolvedModuleLabel_() const override {return realProduct_->moduleLabel();}
    void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
    void setProcessHistory_(ProcessHistory const& ph) override;
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
                             std::vector<bool> const& ambiguous, bool madeAtEnd);

    void connectTo(ProductResolverBase const& iOther, Principal const*) final ;

    void tryPrefetchResolverAsync(unsigned int iProcessingIndex,
                                  Principal const& principal,
                                  bool skipCurrentProcess,
                                  SharedResourcesAcquirer* sra,
                                  ModuleCallingContext const* mcc,
                                  ServiceToken token) const;

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
       void prefetchAsync_(WaitingTask* waitTask,
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

      void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
      void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
      BranchDescription const& branchDescription_() const override;
      void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
      Provenance const* provenance_() const override;

      std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
      void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
      void setProcessHistory_(ProcessHistory const& ph) override;
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
      mutable WaitingTaskList waitingTasks_;
      mutable WaitingTaskList skippingWaitingTasks_;
      mutable std::atomic<unsigned int> lastCheckIndex_;
      mutable std::atomic<unsigned int> lastSkipCurrentCheckIndex_;
      mutable std::atomic<bool> prefetchRequested_;
      mutable std::atomic<bool> skippingPrefetchRequested_;
      const bool madeAtEnd_;
  };

  class SingleChoiceNoProcessProductResolver : public ProductResolverBase {
  public:
    typedef ProducedProductResolver::ProductStatus ProductStatus;
    SingleChoiceNoProcessProductResolver(ProductResolverIndex iChoice):
    ProductResolverBase(), realResolverIndex_(iChoice) {}

    void connectTo(ProductResolverBase const& iOther, Principal const*) final ;

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const override;
     void prefetchAsync_(WaitingTask* waitTask,
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

    void putProduct_(std::unique_ptr<WrapperBase> edp) const override;
    void putOrMergeProduct_(std::unique_ptr<WrapperBase> prod, MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
    BranchDescription const& branchDescription_() const override;
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) override;
    Provenance const* provenance_() const override;

    std::string const& resolvedModuleLabel_() const override {return moduleLabel();}
    void setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) override;
    void setProcessHistory_(ProcessHistory const& ph) override;
    ProductProvenance const* productProvenancePtr_() const override;
    void resetProductData_(bool deleteEarly) override;
    bool singleProduct_() const override;

    ProductResolverIndex realResolverIndex_;
  };

}

#endif
