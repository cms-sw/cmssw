#include "FWCore/Framework/interface/maker/WorkerT.h"
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
    struct has_stream_functions {
      static bool constexpr value = false;
    };

    template <>
    struct has_stream_functions<edm::global::EDProducerBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::global::EDFilterBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::global::EDAnalyzerBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::limited::EDProducerBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::limited::EDFilterBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::limited::EDAnalyzerBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::stream::EDProducerAdaptorBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::stream::EDFilterAdaptorBase> {
      static bool constexpr value = true;
    };

    template <>
    struct has_stream_functions<edm::stream::EDAnalyzerAdaptorBase> {
      static bool constexpr value = true;
    };

    template <typename T>
    struct has_only_stream_transition_functions {
      static bool constexpr value = false;
    };

    template <>
    struct has_only_stream_transition_functions<edm::global::OutputModuleBase> {
      static bool constexpr value = true;
    };

    struct DoNothing {
      template <typename... T>
      inline void operator()(const T&...) {}
    };

    template <typename T, typename TI, typename TP>
    struct DoBeginStream {
      inline void operator()(WorkerT<T, TI, TP>* iWorker, StreamID id) { iWorker->callWorkerBeginStream(0, id); }
    };

    template <typename T, typename TI, typename TP>
    struct DoEndStream {
      inline void operator()(WorkerT<T, TI, TP>* iWorker, StreamID id) { iWorker->callWorkerEndStream(0, id); }
    };

    template <typename T, typename TI, typename TP, typename INFOTYPE>
    struct DoStreamBeginTrans {
      inline void operator()(WorkerT<T, TI, TP>* iWorker,
                             StreamID id,
                             INFOTYPE const& info,
                             ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamBegin(0, id, info, mcc);
      }
    };

    template <typename T, typename TI, typename TP, typename INFOTYPE>
    struct DoStreamEndTrans {
      inline void operator()(WorkerT<T, TI, TP>* iWorker,
                             StreamID id,
                             INFOTYPE const& info,
                             ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamEnd(0, id, info, mcc);
      }
    };
  }  // namespace workerimpl

  template <typename T, typename TI, typename TP>
  WorkerTBase<T, TI, TP>::WorkerTBase(std::shared_ptr<T> ed,
                                      ModuleDescription const& md,
                                      ExceptionToActionTable const* actions)
      : TransitionWorker<TI, TP>(md, actions), module_(ed) {
    assert(module_ != nullptr);
  }

  template <typename T, typename TI, typename TP>
  WorkerT<T, TI, TP>::WorkerT(std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions)
      : WorkerTBase<T, TI, TP>(ed, md, actions) {}

  template <typename T, typename TI>
  WorkerT<T, TI, TransitionPhaseStream>::WorkerT(std::shared_ptr<T> ed,
                                                 ModuleDescription const& md,
                                                 ExceptionToActionTable const* actions)
      : WorkerTBase<T, TI, TransitionPhaseStream>(ed, md, actions) {}

  template <typename T>
  WorkerT<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::WorkerT(std::shared_ptr<T> ed,
                                                                              ModuleDescription const& md,
                                                                              ExceptionToActionTable const* actions)
      : WorkerTBase<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T>
  WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::WorkerT(std::shared_ptr<T> ed,
                                                                         ModuleDescription const& md,
                                                                         ExceptionToActionTable const* actions)
      : WorkerTBase<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T>
  WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::WorkerT(std::shared_ptr<T> ed,
                                                                  ModuleDescription const& md,
                                                                  ExceptionToActionTable const* actions)
      : WorkerTBase<T, EventTransitionInfo, TransitionPhaseGlobal>(ed, md, actions) {}

  template <typename T, typename TI, typename TP>
  bool WorkerT<T, TI, TP>::wantsGlobalTransitions() const noexcept {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      return this->module().wantsGlobalRuns();
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      return this->module().wantsGlobalLuminosityBlocks();
    } else {
      return false;
    }
  }

  template <typename T, typename TI, typename TP>
  bool WorkerT<T, TI, TP>::wantsWrites() const noexcept {
    return false;
  }
#define EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(M, T) \
  M(T, RunTransitionInfo, TransitionPhaseGlobal)     \
  M(T, LumiTransitionInfo, TransitionPhaseGlobal)

#define EDM_SPECIALIZE_WORKERT_WANTS_WRITES(T, TI, TP)    \
  template <>                                             \
  bool WorkerT<T, TI, TP>::wantsWrites() const noexcept { \
    return true;                                          \
  }

  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_WANTS_WRITES, edm::global::OutputModuleBase)
  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_WANTS_WRITES, edm::one::OutputModuleBase)
  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_WANTS_WRITES, edm::limited::OutputModuleBase)

#undef EDM_SPECIALIZE_WORKERT_WANTS_WRITES

  template <typename T, typename TI, typename TP>
  SerialTaskQueue* WorkerT<T, TI, TP>::globalTransitionsQueue() {
    return nullptr;
  }

//one
#define EDM_SPECIALIZE_WORKERT_GLOBALQUEUES(T, TI, TP)             \
  template <>                                                      \
  SerialTaskQueue* WorkerT<T, TI, TP>::globalTransitionsQueue() {  \
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {         \
      return this->module().globalRunsQueue();                     \
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) { \
      return this->module().globalLuminosityBlocksQueue();         \
    }                                                              \
  }

  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_GLOBALQUEUES, one::EDProducerBase)
  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_GLOBALQUEUES, one::EDFilterBase)
  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_GLOBALQUEUES, one::EDAnalyzerBase)
  EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION(EDM_SPECIALIZE_WORKERT_GLOBALQUEUES, one::OutputModuleBase)
#undef EDM_SPECIALIZE_WORKERT_GLOBALQUEUES

  template <typename T>
  bool WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::implDo(EventTransitionInfo const& info,
                                                                      ModuleCallingContext const* mcc) {
    return this->module().doEvent(info, mcc);
  }

  template <typename T>
  void WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoAcquire(EventTransitionInfo const& info,
                                                                             ModuleCallingContext const* mcc,
                                                                             WaitingTaskHolder&& holder) {
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
  void WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoTransformAsync(
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
  size_t WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::transformIndex(
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
  ProductResolverIndex WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::itemToGetForTransform(
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
  bool WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::implNeedToRunSelection() const noexcept {
    return false;
  }

  template <>
  bool WorkerT<edm::one::OutputModuleBase, EventTransitionInfo, TransitionPhaseGlobal>::implNeedToRunSelection()
      const noexcept {
    return true;
  }
  template <>
  bool WorkerT<edm::global::OutputModuleBase, EventTransitionInfo, TransitionPhaseGlobal>::implNeedToRunSelection()
      const noexcept {
    return true;
  }
  template <>
  bool WorkerT<edm::limited::OutputModuleBase, EventTransitionInfo, TransitionPhaseGlobal>::implNeedToRunSelection()
      const noexcept {
    return true;
  }

  template <typename T>
  bool WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::implDoPrePrefetchSelection(
      StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::one::OutputModuleBase,
                                        edm::global::OutputModuleBase,
                                        edm::limited::OutputModuleBase>::value) {
      return this->module().prePrefetchSelection(id, ep, mcc);
    }
    return false;
  }
  template <typename T>
  void WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>::itemsToGetForSelection(
      std::vector<ProductResolverIndexAndSkipBit>& iItems) const {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::one::OutputModuleBase,
                                        edm::global::OutputModuleBase,
                                        edm::limited::OutputModuleBase>::value) {
      iItems = this->module().productsUsedBySelection();
    }
  }

  template <typename T>
  bool WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoBeginProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doBeginProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T>
  bool WorkerT<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoAccessInputProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doAccessInputProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T>
  bool WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>::implDoEndProcessBlock(
      ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
    this->module().doEndProcessBlock(pbp, mcc);
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool WorkerT<T, TI, TP>::implDoBegin(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doBeginRun(info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doBeginLuminosityBlock(info, mcc);
    }
    return true;
  }

  template <typename T, typename TI>
  template <typename D>
  void WorkerT<T, TI, TransitionPhaseStream>::callWorkerStreamBegin(D,
                                                                    StreamID id,
                                                                    TI const& info,
                                                                    ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doStreamBeginRun(id, info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doStreamBeginLuminosityBlock(id, info, mcc);
    }
  }

  template <typename T, typename TI>
  template <typename D>
  void WorkerT<T, TI, TransitionPhaseStream>::callWorkerStreamEnd(D,
                                                                  StreamID id,
                                                                  TI const& info,
                                                                  ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doStreamEndRun(id, info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doStreamEndLuminosityBlock(id, info, mcc);
    }
  }

  template <typename T, typename TI>
  bool WorkerT<T, TI, TransitionPhaseStream>::implDoStreamBegin(StreamID id,
                                                                TI const& info,
                                                                ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      std::conditional_t<workerimpl::has_stream_functions<T>::value,
                         workerimpl::DoStreamBeginTrans<T, TI, TransitionPhaseStream, RunTransitionInfo const>,
                         workerimpl::DoNothing>
          might_call;
      might_call(this, id, info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      std::conditional_t<workerimpl::has_stream_functions<T>::value,
                         workerimpl::DoStreamBeginTrans<T, TI, TransitionPhaseStream, LumiTransitionInfo const>,
                         workerimpl::DoNothing>
          might_call;
      might_call(this, id, info, mcc);
    }
    return true;
  }

  template <typename T, typename TI>
  bool WorkerT<T, TI, TransitionPhaseStream>::implDoStreamEnd(StreamID id,
                                                              TI const& info,
                                                              ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      std::conditional_t<workerimpl::has_stream_functions<T>::value,
                         workerimpl::DoStreamEndTrans<T, TI, TransitionPhaseStream, RunTransitionInfo const>,
                         workerimpl::DoNothing>
          might_call;
      might_call(this, id, info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      std::conditional_t<workerimpl::has_stream_functions<T>::value,
                         workerimpl::DoStreamEndTrans<T, TI, TransitionPhaseStream, LumiTransitionInfo const>,
                         workerimpl::DoNothing>
          might_call;
      might_call(this, id, info, mcc);
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool WorkerT<T, TI, TP>::implDoEnd(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
      this->module().doEndRun(info, mcc);
    } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
      this->module().doEndLuminosityBlock(info, mcc);
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  bool WorkerT<T, TI, TP>::implDoWrite(TI const& info, ModuleCallingContext const* mcc) {
    if constexpr (workerimpl::is_one_of<T,
                                        edm::one::OutputModuleBase,
                                        edm::global::OutputModuleBase,
                                        edm::limited::OutputModuleBase>::value) {
      if constexpr (std::is_same_v<TI, RunTransitionInfo>) {
        this->module().doWriteRun(info.principal(), mcc);
      } else if constexpr (std::is_same_v<TI, LumiTransitionInfo>) {
        this->module().doWriteLuminosityBlock(info.principal(), mcc);
      }
    }
    return true;
  }

  template <typename T, typename TI, typename TP>
  typename Worker::TaskQueueAdaptor WorkerTBase<T, TI, TP>::serializeRunModule() {
    return typename Worker::TaskQueueAdaptor{};
  }
#define EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, TI, TP) \
  template <>                                                                      \
  Worker::TaskQueueAdaptor WorkerTBase<T, TI, TP>::serializeRunModule() {          \
    return QUEUE;                                                                  \
  }                                                                                \
  template <>                                                                      \
  Worker::Types WorkerTBase<T, TI, TP>::moduleType() const {                       \
    return Worker::Types::TYPE;                                                    \
  }                                                                                \
  template <>                                                                      \
  Worker::ConcurrencyTypes WorkerTBase<T, TI, TP>::moduleConcurrencyType() const { \
    return Worker::ConcurrencyTypes::CONCURRENCY;                                  \
  }

#define EDM_SPECIALIZE_WORKERT(T, TYPE, CONCURRENCY, QUEUE)                                                     \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, RunTransitionInfo, TransitionPhaseGlobal)  \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, RunTransitionInfo, TransitionPhaseStream)  \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, LumiTransitionInfo, TransitionPhaseGlobal) \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, LumiTransitionInfo, TransitionPhaseStream) \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(                                                                        \
      T, TYPE, CONCURRENCY, QUEUE, ProcessBlockTransitionInfo, TransitionPhaseGlobal)                           \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(                                                                        \
      T, TYPE, CONCURRENCY, QUEUE, InputProcessBlockTransitionInfo, TransitionPhaseGlobal)                      \
  EDM_SPECIALIZE_WORKERT_FOR_TRANSITION(T, TYPE, CONCURRENCY, QUEUE, EventTransitionInfo, TransitionPhaseGlobal)

  EDM_SPECIALIZE_WORKERT(one::EDProducerBase,
                         kProducer,
                         kOne,
                         &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_WORKERT(one::EDFilterBase,
                         kFilter,
                         kOne,
                         &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_WORKERT(one::EDAnalyzerBase,
                         kAnalyzer,
                         kOne,
                         &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_WORKERT(one::OutputModuleBase,
                         kOutputModule,
                         kOne,
                         &(this->module().sharedResourcesAcquirer().serialQueueChain()))
  EDM_SPECIALIZE_WORKERT(global::EDProducerBase, kProducer, kGlobal, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(global::EDFilterBase, kFilter, kGlobal, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(global::EDAnalyzerBase, kAnalyzer, kGlobal, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(global::OutputModuleBase, kOutputModule, kGlobal, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(limited::EDProducerBase, kProducer, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_WORKERT(limited::EDFilterBase, kFilter, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_WORKERT(limited::EDAnalyzerBase, kAnalyzer, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_WORKERT(limited::OutputModuleBase, kOutputModule, kLimited, &(this->module().queue()))
  EDM_SPECIALIZE_WORKERT(stream::EDProducerAdaptorBase, kProducer, kStream, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(stream::EDFilterAdaptorBase, kFilter, kStream, Worker::TaskQueueAdaptor{})
  EDM_SPECIALIZE_WORKERT(stream::EDAnalyzerAdaptorBase, kAnalyzer, kStream, Worker::TaskQueueAdaptor{})

#undef EDM_SPECIALIZE_WORKERT
#undef EDM_SPECIALIZE_WORKERT_FOR_TRANSITION

  //Explicitly instantiate our needed templates to avoid having the compiler
  // instantiate them in all of our libraries
#define EDM_INSTANTIATE_WORKERT(T)                                                   \
  template class WorkerT<T, RunTransitionInfo, TransitionPhaseGlobal>;               \
  template class WorkerT<T, RunTransitionInfo, TransitionPhaseStream>;               \
  template class WorkerT<T, LumiTransitionInfo, TransitionPhaseGlobal>;              \
  template class WorkerT<T, LumiTransitionInfo, TransitionPhaseStream>;              \
  template class WorkerT<T, ProcessBlockTransitionInfo, TransitionPhaseGlobal>;      \
  template class WorkerT<T, InputProcessBlockTransitionInfo, TransitionPhaseGlobal>; \
  template class WorkerT<T, EventTransitionInfo, TransitionPhaseGlobal>;

  EDM_INSTANTIATE_WORKERT(one::EDProducerBase)
  EDM_INSTANTIATE_WORKERT(one::EDFilterBase)
  EDM_INSTANTIATE_WORKERT(one::EDAnalyzerBase)
  EDM_INSTANTIATE_WORKERT(one::OutputModuleBase)
  EDM_INSTANTIATE_WORKERT(global::EDProducerBase)
  EDM_INSTANTIATE_WORKERT(global::EDFilterBase)
  EDM_INSTANTIATE_WORKERT(global::EDAnalyzerBase)
  EDM_INSTANTIATE_WORKERT(global::OutputModuleBase)
  EDM_INSTANTIATE_WORKERT(stream::EDProducerAdaptorBase)
  EDM_INSTANTIATE_WORKERT(stream::EDFilterAdaptorBase)
  EDM_INSTANTIATE_WORKERT(stream::EDAnalyzerAdaptorBase)
  EDM_INSTANTIATE_WORKERT(limited::EDProducerBase)
  EDM_INSTANTIATE_WORKERT(limited::EDFilterBase)
  EDM_INSTANTIATE_WORKERT(limited::EDAnalyzerBase)
  EDM_INSTANTIATE_WORKERT(limited::OutputModuleBase)

#undef EDM_INSTANTIATE_WORKERT
#undef EDM_FOR_EACH_WORKERT_GLOBAL_TRANSITION
}  // namespace edm
