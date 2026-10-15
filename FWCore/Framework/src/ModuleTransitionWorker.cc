#include "FWCore/Framework/interface/maker/ModuleTransitionWorker.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/one/EDFilterBase.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"
#include "FWCore/Framework/interface/global/EDFilterBase.h"
#include "FWCore/Framework/interface/global/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/global/OutputModuleBase.h"

#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"

#include "FWCore/Framework/interface/limited/EDProducerBase.h"
#include "FWCore/Framework/interface/limited/EDFilterBase.h"
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"

#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"

#include <type_traits>

namespace edm {
  namespace workerimpl {
    template <typename T, typename... U>
    struct is_one_of {
      static bool constexpr value = (std::is_base_of_v<U, T> || ...);
    };

    template <typename T>
    using is_outputmodule =
        is_one_of<T, edm::one::OutputModuleBase, edm::global::OutputModuleBase, edm::limited::OutputModuleBase>;

    template <typename T>
    using is_one_module = is_one_of<T,
                                    edm::one::EDProducerBase,
                                    edm::one::EDFilterBase,
                                    edm::one::EDAnalyzerBase,
                                    edm::one::OutputModuleBase>;
    template <typename T>
    struct has_stream_functions {
      static bool constexpr value = not(is_one_module<T>::value or is_outputmodule<T>::value);
    };

  }  // namespace workerimpl

  template <typename T, typename TI, typename TP>
  ModuleTransitionWorkerBase<T, TI, TP>::ModuleTransitionWorkerBase(std::shared_ptr<T> ed,
                                                                    ModuleDescription const& md,
                                                                    ExceptionToActionTable const* actions)
      : TransitionWorker<TI, TP>(md, actions), module_(ed) {
    assert(module_ != nullptr);
  }

  template <typename T, typename TI, typename TP>
  ModuleTransitionWorker<T, TI, TP>::ModuleTransitionWorker(std::shared_ptr<T> ed,
                                                            ModuleDescription const& md,
                                                            ExceptionToActionTable const* actions)
      : ModuleTransitionWorkerBase<T, TI, TP>(ed, md, actions) {}

  template <typename T, typename TI>
  ModuleTransitionWorker<T, TI, TransitionPhaseStream>::ModuleTransitionWorker(std::shared_ptr<T> ed,
                                                                               ModuleDescription const& md,
                                                                               ExceptionToActionTable const* actions)
      : ModuleTransitionWorkerBase<T, TI, TransitionPhaseStream>(ed, md, actions) {}

  template <typename T>
  ModuleTransitionWorker<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::ModuleTransitionWorker(
      std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions)
      : ModuleTransitionWorkerBase<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T>
  ModuleTransitionWorker<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::ModuleTransitionWorker(
      std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions)
      : ModuleTransitionWorkerBase<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T>
  ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::ModuleTransitionWorker(
      std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions)
      : ModuleTransitionWorkerBase<T, EventTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T, typename TI, typename TP>
  bool ModuleTransitionWorker<T, TI, TP>::wantsGlobalTransitions() const noexcept {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      return this->module().wantsGlobalRuns();
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      return this->module().wantsGlobalLuminosityBlocks();
    } else {
      return false;
    }
  }

  template <typename T, typename TI, typename TP>
  bool ModuleTransitionWorker<T, TI, TP>::wantsWrites() const noexcept {
    if constexpr (workerimpl::is_outputmodule<T>::value) {
      return true;
    }
    return false;
  }

  template <typename T, typename TI, typename TP>
  SerialTaskQueue* ModuleTransitionWorker<T, TI, TP>::globalTransitionsQueue() {
    //ones are special
    if constexpr (workerimpl::is_one_module<T>::value) {
      if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
        return this->module().globalRunsQueue();
      } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
        return this->module().globalLuminosityBlocksQueue();
      }
    }
    return nullptr;
  }

  template <typename T>
  bool ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::implDo(EventTransitionInfo const& info,
                                                                                     ModuleCallingContext const* mcc) {
    return this->module().doEvent(info, mcc);
  }

  template <typename T>
  void ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoAcquire(
      EventTransitionInfo const& info, ModuleCallingContext const* mcc, WaitingTaskHolder&& holder) {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::global::EDProducerBase,
                                        edm::global::EDFilterBase,
                                        edm::global::OutputModuleBase,
                                        edm::stream::EDProducerAdaptorBase,
                                        edm::stream::EDFilterAdaptorBase>::value) {
      this->module().doAcquire(info, mcc, std::move(holder));
    }
  }

  template <typename T>
  void ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoTransformAsync(
      WaitingTaskHolder iTask,
      size_t iTransformIndex,
      EventPrincipal const& iEvent,
      ParentContext const& iParent,
      ServiceWeakToken const& weakToken) noexcept {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::global::EDProducerBase,
                                        edm::global::EDFilterBase,
                                        edm::limited::EDProducerBase,
                                        edm::limited::EDFilterBase,
                                        edm::one::EDProducerBase,
                                        edm::one::EDFilterBase,
                                        edm::stream::EDProducerAdaptorBase,
                                        edm::stream::EDFilterAdaptorBase>::value) {
      CMS_SA_ALLOW try {
        ServiceRegistry::Operate guard(weakToken.lock());

        ModuleCallingContext mcc(&this->module().moduleDescription(),
                                 iTransformIndex + 1,
                                 ModuleCallingContext::State::kRunning,
                                 iParent,
                                 nullptr);
        this->module().doTransformAsync(iTask, iTransformIndex, iEvent, this->activityRegistry(), mcc, weakToken);
      } catch (...) {
        iTask.doneWaiting(std::current_exception());
        return;
      }
      iTask.doneWaiting(std::exception_ptr());
    }
  }

  template <typename T>
  size_t ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::transformIndex(
      edm::ProductDescription const& iBranch) const noexcept {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::global::EDProducerBase,
                                        edm::global::EDFilterBase,
                                        edm::limited::EDProducerBase,
                                        edm::limited::EDFilterBase,
                                        edm::one::EDProducerBase,
                                        edm::one::EDFilterBase,
                                        edm::stream::EDProducerAdaptorBase>::value) {
      return this->module().transformIndex_(iBranch);
    }
    return 0;
  }

  template <typename T>
  ProductResolverIndex ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::itemToGetForTransform(
      size_t iTransformIndex) const noexcept {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::global::EDProducerBase,
                                        edm::global::EDFilterBase,
                                        edm::limited::EDProducerBase,
                                        edm::limited::EDFilterBase,
                                        edm::one::EDProducerBase,
                                        edm::one::EDFilterBase,
                                        edm::stream::EDProducerAdaptorBase>::value) {
      return this->module().transformPrefetch_(iTransformIndex);
    }
    return 0;
  }

  template <typename T>
  bool ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::implNeedToRunSelection() const noexcept {
    if constexpr (workerimpl::is_outputmodule<T>::value) {
      return true;
    }
    return false;
  }

  template <typename T>
  bool ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoPrePrefetchSelection(
      StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::is_outputmodule<T>::value) {
      return this->module().prePrefetchSelection(id, ep, mcc);
    }
    return false;
  }
  template <typename T>
  void ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>::itemsToGetForSelection(
      std::vector<ProductResolverIndexAndSkipBit>& iItems) const {
    if constexpr (workerimpl::is_outputmodule<T>::value) {
      iItems = this->module().productsUsedBySelection();
    }
  }

  template <typename T>
  bool ModuleTransitionWorker<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoBeginProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doBeginProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T>
  bool ModuleTransitionWorker<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoAccessInputProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doAccessInputProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T>
  bool ModuleTransitionWorker<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoEndProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doEndProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool ModuleTransitionWorker<T, TI, TP>::implDoBegin(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doBeginRun(info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doBeginLuminosityBlock(info, mcc);
    }
    return true;
  }

  template <typename T, typename TI>
  bool ModuleTransitionWorker<T, TI, TransitionPhaseStream>::implDoStreamBegin(StreamID id,
                                                                               TI const& info,
                                                                               ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::has_stream_functions<T>::value) {
      if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
        this->module().doStreamBeginRun(id, info, mcc);
      } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
        this->module().doStreamBeginLuminosityBlock(id, info, mcc);
      }
    }
    return true;
  }

  template <typename T, typename TI>
  bool ModuleTransitionWorker<T, TI, TransitionPhaseStream>::implDoStreamEnd(StreamID id,
                                                                             TI const& info,
                                                                             ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::has_stream_functions<T>::value) {
      if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
        this->module().doStreamEndRun(id, info, mcc);
      } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
        this->module().doStreamEndLuminosityBlock(id, info, mcc);
      }
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool ModuleTransitionWorker<T, TI, TP>::implDoEnd(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doEndRun(info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doEndLuminosityBlock(info, mcc);
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool ModuleTransitionWorker<T, TI, TP>::implDoWrite(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::is_outputmodule<T>::value) {
      if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
        this->module().doWriteRun(info.principal(), mcc);
      } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
        this->module().doWriteLuminosityBlock(info.principal(), mcc);
      }
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  typename TransitionWorkerBase::TaskQueueAdaptor ModuleTransitionWorkerBase<T, TI, TP>::serializeRunModule() {
    return typename TransitionWorkerBase::TaskQueueAdaptor{};
  }
#define EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, TI, TP)                     \
  template <>                                                                                                   \
  TransitionWorkerBase::TaskQueueAdaptor ModuleTransitionWorkerBase<T, TI, TP>::serializeRunModule() {          \
    return QUEUE;                                                                                               \
  }                                                                                                             \
  template <>                                                                                                   \
  TransitionWorkerBase::Types ModuleTransitionWorkerBase<T, TI, TP>::moduleType() const {                       \
    return TransitionWorkerBase::Types::TYPE;                                                                   \
  }                                                                                                             \
  template <>                                                                                                   \
  TransitionWorkerBase::ConcurrencyTypes ModuleTransitionWorkerBase<T, TI, TP>::moduleConcurrencyType() const { \
    return TransitionWorkerBase::ConcurrencyTypes::CONCURRENCY;                                                 \
  }

#define EDM_SPECIALIZE_TRANSITIONWORKER(T, TYPE, CONCURRENCY, QUEUE)                       \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, RunTransitionInfo, TransitionPhaseGlobal)               \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, RunTransitionInfo, TransitionPhaseStream)               \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, LumiTransitionInfo, TransitionPhaseGlobal)              \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, LumiTransitionInfo, TransitionPhaseStream)              \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, ProcessBlockTransitionInfo, TransitionPhaseGlobal)      \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, InputProcessBlockTransitionInfo, TransitionPhaseGlobal) \
  EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION(                                          \
      T, TYPE, CONCURRENCY, QUEUE, EventTransitionInfo, TransitionPhaseGlobal)

  EDM_SPECIALIZE_TRANSITIONWORKER(one::EDProducerBase,
                                  kProducer,
                                  kOne,
                                  &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_TRANSITIONWORKER(one::EDFilterBase,
                                  kFilter,
                                  kOne,
                                  &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_TRANSITIONWORKER(one::EDAnalyzerBase,
                                  kAnalyzer,
                                  kOne,
                                  &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_TRANSITIONWORKER(one::OutputModuleBase,
                                  kOutputModule,
                                  kOne,
                                  &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_TRANSITIONWORKER(global::EDProducerBase, kProducer, kGlobal, TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(global::EDFilterBase, kFilter, kGlobal, TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(global::EDAnalyzerBase, kAnalyzer, kGlobal, TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(global::OutputModuleBase,
                                  kOutputModule,
                                  kGlobal,
                                  TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(limited::EDProducerBase, kProducer, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_TRANSITIONWORKER(limited::EDFilterBase, kFilter, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_TRANSITIONWORKER(limited::EDAnalyzerBase, kAnalyzer, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_TRANSITIONWORKER(limited::OutputModuleBase, kOutputModule, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_TRANSITIONWORKER(stream::EDProducerAdaptorBase,
                                  kProducer,
                                  kStream,
                                  TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(stream::EDFilterAdaptorBase,
                                  kFilter,
                                  kStream,
                                  TransitionWorkerBase::TaskQueueAdaptor{})
  EDM_SPECIALIZE_TRANSITIONWORKER(stream::EDAnalyzerAdaptorBase,
                                  kAnalyzer,
                                  kStream,
                                  TransitionWorkerBase::TaskQueueAdaptor{})

#undef EDM_SPECIALIZE_TRANSITIONWORKER
#undef EDM_SPECIALIZE_TRANSITIONWORKER_FOR_TRANSITION

  //Explicitly instantiate our needed templates to avoid having the compiler
  // instantiate them in all of our libraries
#define EDM_INSTANTIATE_TRANSITIONWORKER(T)                                                         \
  template class ModuleTransitionWorker<T, RunTransitionInfo, TransitionPhaseGlobal>;               \
  template class ModuleTransitionWorker<T, RunTransitionInfo, TransitionPhaseStream>;               \
  template class ModuleTransitionWorker<T, LumiTransitionInfo, TransitionPhaseGlobal>;              \
  template class ModuleTransitionWorker<T, LumiTransitionInfo, TransitionPhaseStream>;              \
  template class ModuleTransitionWorker<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>;      \
  template class ModuleTransitionWorker<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>; \
  template class ModuleTransitionWorker<T, EventTransitionInfo, TransitionPhaseGlobal>;

  EDM_INSTANTIATE_TRANSITIONWORKER(one::EDProducerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(one::EDFilterBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(one::EDAnalyzerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(one::OutputModuleBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(global::EDProducerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(global::EDFilterBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(global::EDAnalyzerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(global::OutputModuleBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(stream::EDProducerAdaptorBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(stream::EDFilterAdaptorBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(stream::EDAnalyzerAdaptorBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(limited::EDProducerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(limited::EDFilterBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(limited::EDAnalyzerBase)
  EDM_INSTANTIATE_TRANSITIONWORKER(limited::OutputModuleBase)

#undef EDM_INSTANTIATE_TRANSITIONWORKER
}  // namespace edm
