#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------

WorkerT: Code common to all workers.

----------------------------------------------------------------------*/

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  class ProductResolverIndexAndSkipBit;
  class ThinnedAssociationsHelper;

  namespace eventsetup {
    struct ComponentDescription;
  }  // namespace eventsetup

  template <typename T>
  class WorkerT : public Worker {
  public:
    typedef T ModuleType;
    typedef WorkerT<T> WorkerType;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    ~WorkerT() override;

    void setModule(std::shared_ptr<T> iModule) {
      module_ = iModule;
      resetModuleDescription(&(module_->moduleDescription()));
    }

    Types moduleType() const override;
    ConcurrencyTypes moduleConcurrencyType() const override;

    bool wantsProcessBlocks() const noexcept final;
    bool wantsInputProcessBlocks() const noexcept final;
    bool wantsGlobalRuns() const noexcept final;
    bool wantsGlobalLuminosityBlocks() const noexcept final;
    bool wantsStreamRuns() const noexcept final;
    bool wantsStreamLuminosityBlocks() const noexcept final;

    SerialTaskQueue* globalRunsQueue() final;
    SerialTaskQueue* globalLuminosityBlocksQueue() final;

    template <typename D>
    void callWorkerBeginStream(D, StreamID);
    template <typename D>
    void callWorkerEndStream(D, StreamID);
    template <typename D>
    void callWorkerStreamBegin(D, StreamID, RunTransitionInfo const&, ModuleCallingContext const*);
    template <typename D>
    void callWorkerStreamEnd(D, StreamID, RunTransitionInfo const&, ModuleCallingContext const*);
    template <typename D>
    void callWorkerStreamBegin(D, StreamID, LumiTransitionInfo const&, ModuleCallingContext const*);
    template <typename D>
    void callWorkerStreamEnd(D, StreamID, LumiTransitionInfo const&, ModuleCallingContext const*);

    bool matchesBaseClassPointer(void const* iPtr) const noexcept final { return &(*module_) == iPtr; }

  protected:
    T& module() { return *module_; }
    T const& module() const { return *module_; }

  private:
    void doClearModule() override { get_underlying_safe(module_).reset(); }

    bool implDo(EventTransitionInfo const&, ModuleCallingContext const*) override;

    void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const final;
    bool implNeedToRunSelection() const noexcept final;

    void implDoAcquire(EventTransitionInfo const&, ModuleCallingContext const*, WaitingTaskHolder&&) final;

    size_t transformIndex(edm::ProductDescription const&) const noexcept final;
    void implDoTransformAsync(WaitingTaskHolder,
                              size_t iTransformIndex,
                              EventPrincipal const&,
                              ParentContext const&,
                              ServiceWeakToken const&) noexcept final;
    ProductResolverIndex itemToGetForTransform(size_t iTransformIndex) const noexcept final;

    bool implDoPrePrefetchSelection(StreamID, EventPrincipal const&, ModuleCallingContext const*) override;
    bool implDoBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    bool implDoAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    bool implDoEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    bool implDoBegin(RunTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoStreamBegin(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoStreamEnd(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoEnd(RunTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoBegin(LumiTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoStreamBegin(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoStreamEnd(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) override;
    bool implDoEnd(LumiTransitionInfo const&, ModuleCallingContext const*) override;
    void implRespondToOpenInputFile(FileBlock const& fb) override;
    void implRespondToCloseInputFile(FileBlock const& fb) override;
    void implRespondToCloseOutputFile() override;
    TaskQueueAdaptor serializeRunModule() override;

    std::vector<ModuleConsumesInfo> moduleConsumesInfos() const override;
    std::vector<ModuleConsumesMinimalESInfo> moduleConsumesMinimalESInfos() const final {
      return module_->moduleConsumesMinimalESInfos();
    }

    void itemsToGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsToGet(branchType, indexes);
    }

    void itemsMayGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsMayGet(branchType, indexes);
    }

    std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType iType) const final {
      return module_->itemsToGetFrom(iType);
    }

    std::vector<ESResolverIndex> const& esItemsToGetFrom(Transition iTransition) const override {
      return module_->esGetTokenIndicesVector(iTransition);
    }
    std::vector<ESRecordIndex> const& esRecordsToGetFrom(Transition iTransition) const override {
      return module_->esGetTokenRecordIndicesVector(iTransition);
    }

    void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                      ModuleCallingContext const& iModuleCallingContext,
                                      Principal const& iPrincipal) const noexcept override {
      module_->preActionBeforeRunEventAsync(iTask, iModuleCallingContext, iPrincipal);
    }

    bool hasAcquire() const noexcept override { return module_->hasAcquire(); }

    bool hasAccumulator() const noexcept override { return module_->hasAccumulator(); }

    edm::propagate_const<std::shared_ptr<T>> module_;
  };

}  // namespace edm

#endif
