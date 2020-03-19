#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------

WorkerT: Code common to all workers.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  class ModuleCallingContext;
  class ModuleDescription;
  class ProductResolverIndexAndSkipBit;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTaskWithArenaHolder;

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

    bool wantsGlobalRuns() const final;
    bool wantsGlobalLuminosityBlocks() const final;
    bool wantsStreamRuns() const final;
    bool wantsStreamLuminosityBlocks() const final;

    SerialTaskQueue* globalRunsQueue() final;
    SerialTaskQueue* globalLuminosityBlocksQueue() final;

    void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&) final;
    void updateLookup(eventsetup::ESRecordsToProxyIndices const&) final;
    void resolvePutIndicies(
        BranchType iBranchType,
        std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
            iIndicies) final;

    template <typename D>
    void callWorkerBeginStream(D, StreamID);
    template <typename D>
    void callWorkerEndStream(D, StreamID);
    template <typename D>
    void callWorkerStreamBegin(
        D, StreamID id, RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);
    template <typename D>
    void callWorkerStreamEnd(
        D, StreamID id, RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);
    template <typename D>
    void callWorkerStreamBegin(
        D, StreamID id, LuminosityBlockPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);
    template <typename D>
    void callWorkerStreamEnd(
        D, StreamID id, LuminosityBlockPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);

  protected:
    T& module() { return *module_; }
    T const& module() const { return *module_; }

  private:
    bool implDo(EventPrincipal const& ep, EventSetupImpl const& c, ModuleCallingContext const* mcc) override;

    void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const final;
    bool implNeedToRunSelection() const final;

    void implDoAcquire(EventPrincipal const& ep,
                       EventSetupImpl const& c,
                       ModuleCallingContext const* mcc,
                       WaitingTaskWithArenaHolder& holder) final;

    bool implDoPrePrefetchSelection(StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) override;
    bool implDoBegin(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) override;
    bool implDoStreamBegin(StreamID id,
                           RunPrincipal const& rp,
                           EventSetupImpl const& c,
                           ModuleCallingContext const* mcc) override;
    bool implDoStreamEnd(StreamID id,
                         RunPrincipal const& rp,
                         EventSetupImpl const& c,
                         ModuleCallingContext const* mcc) override;
    bool implDoEnd(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) override;
    bool implDoBegin(LuminosityBlockPrincipal const& lbp,
                     EventSetupImpl const& c,
                     ModuleCallingContext const* mcc) override;
    bool implDoStreamBegin(StreamID id,
                           LuminosityBlockPrincipal const& lbp,
                           EventSetupImpl const& c,
                           ModuleCallingContext const* mcc) override;
    bool implDoStreamEnd(StreamID id,
                         LuminosityBlockPrincipal const& lbp,
                         EventSetupImpl const& c,
                         ModuleCallingContext const* mcc) override;
    bool implDoEnd(LuminosityBlockPrincipal const& lbp,
                   EventSetupImpl const& c,
                   ModuleCallingContext const* mcc) override;
    void implBeginJob() override;
    void implEndJob() override;
    void implBeginStream(StreamID) override;
    void implEndStream(StreamID) override;
    void implRespondToOpenInputFile(FileBlock const& fb) override;
    void implRespondToCloseInputFile(FileBlock const& fb) override;
    void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) override;
    std::string workerType() const override;
    TaskQueueAdaptor serializeRunModule() override;

    void modulesWhoseProductsAreConsumed(
        std::vector<ModuleDescription const*>& modules,
        ProductRegistry const& preg,
        std::map<std::string, ModuleDescription const*> const& labelsToDesc) const override {
      module_->modulesWhoseProductsAreConsumed(modules, preg, labelsToDesc, module_->moduleDescription().processName());
    }

    void convertCurrentProcessAlias(std::string const& processName) override {
      module_->convertCurrentProcessAlias(processName);
    }

    std::vector<ConsumesInfo> consumesInfo() const override { return module_->consumesInfo(); }

    void itemsToGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsToGet(branchType, indexes);
    }

    void itemsMayGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsMayGet(branchType, indexes);
    }

    std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType iType) const final {
      return module_->itemsToGetFrom(iType);
    }

    std::vector<ProductResolverIndex> const& itemsShouldPutInEvent() const override;

    void preActionBeforeRunEventAsync(WaitingTask* iTask,
                                      ModuleCallingContext const& iModuleCallingContext,
                                      Principal const& iPrincipal) const override {
      module_->preActionBeforeRunEventAsync(iTask, iModuleCallingContext, iPrincipal);
    }

    bool hasAcquire() const override { return module_->hasAcquire(); }

    bool hasAccumulator() const override { return module_->hasAccumulator(); }

    edm::propagate_const<std::shared_ptr<T>> module_;
  };

}  // namespace edm

#endif
