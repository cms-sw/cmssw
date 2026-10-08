#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------

WorkerT: Code common to all workers.

----------------------------------------------------------------------*/

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/Framework/interface/maker/TransitionWorker.h"
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

  namespace eventsetup {
    struct ComponentDescription;
  }  // namespace eventsetup

  template <typename T, typename TI, typename TP>
  class WorkerTBase : public TransitionWorker<TI, TP> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerTBase<T, TI, TP>;
    using Base = TransitionWorker<TI, TP>;

    WorkerTBase(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    void setModule(std::shared_ptr<T> iModule) {
      module_ = iModule;
      this->resetModuleDescription(&(module_->moduleDescription()));
    }

    Base::Types moduleType() const override;
    Base::ConcurrencyTypes moduleConcurrencyType() const override;

    bool matchesBaseClassPointer(void const* iPtr) const noexcept final { return &(*module_) == iPtr; }

  protected:
    T& module() { return *module_; }
    T const& module() const { return *module_; }

    void doClearModule() override { get_underlying_safe(module_).reset(); }

    TransitionWorkerBase::TaskQueueAdaptor serializeRunModule() override;

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

  private:
    edm::propagate_const<std::shared_ptr<T>> module_;
  };
  template <typename T, typename TI, typename TP>
  class WorkerT : public WorkerTBase<T, TI, TP> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerT<T, TI, TP>;
    using Base = TransitionWorker<TI, TP>;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    using Base::moduleConcurrencyType;
    using Base::moduleType;

    bool wantsGlobalTransitions() const noexcept final;
    bool wantsWrites() const noexcept final;

    SerialTaskQueue* globalTransitionsQueue() final;

  private:
    bool implDoBegin(TI const&, ModuleCallingContext const*) override;
    bool implDoEnd(TI const&, ModuleCallingContext const*) override;
    bool implDoWrite(TI const&, ModuleCallingContext const*) override;
    using Base::serializeRunModule;
  };

  template <typename T, typename TI>
  class WorkerT<T, TI, TransitionPhaseStream> : public WorkerTBase<T, TI, TransitionPhaseStream> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerT<T, TI, TransitionPhaseStream>;
    using Base = TransitionWorker<TI, TransitionPhaseStream>;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    using Base::moduleConcurrencyType;
    using Base::moduleType;

  private:
    bool implDoStreamBegin(StreamID, TI const&, ModuleCallingContext const*) override;
    bool implDoStreamEnd(StreamID, TI const&, ModuleCallingContext const*) override;
    using Base::serializeRunModule;
  };

  template <typename T>
  class WorkerT<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>
      : public WorkerTBase<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerT<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>;
    using Base = TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    using Base::moduleConcurrencyType;
    using Base::moduleType;

  private:
    bool implDoAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    using Base::serializeRunModule;
  };

  template <typename T>
  class WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>
      : public WorkerTBase<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>;
    using Base = TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    using Base::moduleConcurrencyType;
    using Base::moduleType;

  private:
    bool implDoBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    bool implDoEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) override;
    using Base::serializeRunModule;
  };

  template <typename T>
  class WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>
      : public WorkerTBase<T, EventTransitionInfo, TransitionPhaseGlobal> {
  public:
    using ModuleType = T;
    using WorkerType = WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>;
    using Base = TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>;
    WorkerT(std::shared_ptr<T>, ModuleDescription const&, ExceptionToActionTable const* actions);

    using Base::moduleConcurrencyType;
    using Base::moduleType;

  private:
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
    using Base::serializeRunModule;

    void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                      ModuleCallingContext const& iModuleCallingContext,
                                      Principal const& iPrincipal) const noexcept override {
      this->module().preActionBeforeRunEventAsync(iTask, iModuleCallingContext, iPrincipal);
    }

    bool hasAcquire() const noexcept override { return this->module().hasAcquire(); }

    bool hasAccumulator() const noexcept override { return this->module().hasAccumulator(); }
  };

}  // namespace edm

#endif
