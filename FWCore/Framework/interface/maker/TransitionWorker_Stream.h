#ifndef FWCore_Framework_maker_TransitionWorker_Common_h
#define FWCore_Framework_maker_TransitionWorker_Common_h

/*----------------------------------------------------------------------

Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

A worker will not actually call through to the module unless it is
in a Ready state.  After a module is actually run, the state will not
be Ready.  The Ready state can only be reestablished by doing a reset().

Pre/post module signals are posted only in the Ready state.

Execution statistics are kept here.

If a module has thrown an exception during execution, that exception
will be rethrown if the worker is entered again and the state is not Ready.
In other words, execution results (status) are cached and reused until
the worker is reset().

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/Framework/interface/maker/ModuleSignalSentry.h"
#include "FWCore/Framework/interface/maker/ModuleAttributes.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <array>
#include <atomic>
#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>

namespace edm {
  class EventPrincipal;
  class EventSetupImpl;
  class EarlyDeleteHelper;
  class ProductResolverIndexAndSkipBit;

  namespace workerhelper {
    template <typename O>
    class CallImpl;
  }
  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  class EventTransitionInfo;
  class RunTransitionInfo;
  class LumiTransitionInfo;
  struct TransitionPhaseGlobal;
  struct TransitionPhaseStream;

  template <typename TI, TransitionActionType T>
  struct TransitionActionContextTrait;
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionStreamBegin> {
    using ContextType = StreamContext;
  };

  template <typename TI, typename TP, TransitionEdge E>
  struct TransitionActionTrait;
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseStream, TransitionEdge::kBegin> {
    static constexpr TransitionActionType value = TransitionActionStreamBegin;
  };
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseStream, TransitionEdge::kEnd> {
    static constexpr TransitionActionType value = TransitionActionStreamEnd;
  };

  template <typename TI>
  class TransitionWorker<TI, TransitionPhaseStream> : public Worker {
  public:
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}

    virtual bool wantsStreamRuns() const noexcept = 0;
    virtual bool wantsStreamLuminosityBlocks() const noexcept = 0;

    void reset() { resetBase(); }

    template <TransitionEdge E>
    void doWorkAsync(
        WaitingTaskHolder iTask,
        TI const& iTransitionInfo,
        ServiceToken const& iToken,
        StreamID iStreamID,
        ParentContext const& iParentContext,
        typename TransitionActionContextTrait<TI, TransitionActionTrait<TI, TP, E>::value>::ContextType const*
            iContext) noexcept {
      this->template doWorkAsyncImpl<
          OccurrenceTraits<typename TI::PrincipalType, TransitionActionTrait<TI, TP, E>::value>>(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoPrefetchingAsync(
        WaitingTaskHolder iTask,
        TI const& iTransitionInfo,
        ServiceToken const& iToken,
        StreamID iStreamID,
        ParentContext const& iParentContext,
        typename TransitionActionContextTrait<TI, TransitionActionTrait<TI, TP, E>::value>::ContextType const*
            iContext) noexcept {
      this->template doWorkNoPrefetchingAsyncImpl<
          OccurrenceTraits<typename TI::PrincipalType, TransitionActionTrait<TI, TP, E>::value>>(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

  protected:
    //Called by GlobalSchedule::processOneGlobalAsync, UnscheduledCallProducer::runAccumulatorsAsync, WorkerInPath::runWorkerAsync, UnscheduledProductResolver::prefetchAsync_
    template <typename T>
    void doWorkAsyncImpl(WaitingTaskHolder,
                         typename T::TransitionInfoType const&,
                         ServiceToken const&,
                         StreamID,
                         ParentContext const&,
                         typename T::Context const*) noexcept;

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <typename T>
    void doWorkNoPrefetchingAsyncImpl(WaitingTaskHolder,
                                      typename T::TransitionInfoType const&,
                                      ServiceToken const&,
                                      StreamID,
                                      ParentContext const&,
                                      typename T::Context const*) noexcept;

    template <typename O>
    friend class workerhelper::CallImpl;

    virtual bool implDoStreamBegin(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamBegin(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;

  private:
    template <typename T>
    bool runModule(typename T::TransitionInfoType const&, StreamID, ParentContext const&, typename T::Context const*);

    template <typename T>
    void prefetchAsync(WaitingTaskHolder,
                       ServiceToken const&,
                       ParentContext const&,
                       typename T::TransitionInfoType const&,
                       Transition) noexcept;

    void emitPostModuleStreamPrefetchingSignal() {
      actReg_->postModuleStreamPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(),
                                                       moduleCallingContext_);
    }

    void emitPostModuleGlobalPrefetchingSignal() {
      actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(),
                                                       moduleCallingContext_);
    }

    template <typename T>
    std::exception_ptr runModuleAfterAsyncPrefetch(std::exception_ptr,
                                                   typename T::TransitionInfoType const&,
                                                   StreamID,
                                                   ParentContext const&,
                                                   typename T::Context const*) noexcept;

    std::exception_ptr handleExternalWorkException(std::exception_ptr iEPtr,
                                                   ParentContext const& parentContext) noexcept;

    template <typename T>
    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(TransitionWorker<TI, TP>* worker,
                    typename T::TransitionInfoType const& transitionInfo,
                    ServiceToken const& token,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    typename T::Context const* context,
                    oneapi::tbb::task_group* iGroup) noexcept
          : m_worker(worker),
            m_transitionInfo(transitionInfo),
            m_streamID(streamID),
            m_parentContext(parentContext),
            m_context(context),
            m_serviceToken(token),
            m_group(iGroup) {}

      struct EnableQueueGuard {
        SerialTaskQueue* queue_;
        EnableQueueGuard(SerialTaskQueue* iQueue) : queue_{iQueue} {}
        EnableQueueGuard(EnableQueueGuard const&) = delete;
        EnableQueueGuard& operator=(EnableQueueGuard const&) = delete;
        EnableQueueGuard& operator=(EnableQueueGuard&&) = delete;
        EnableQueueGuard(EnableQueueGuard&& iGuard) : queue_{iGuard.queue_} { iGuard.queue_ = nullptr; }
        ~EnableQueueGuard() {
          if (queue_) {
            queue_->resume();
          }
        }
      };

      void execute() final {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken.lock());

        //incase the emit causes an exception, we need a memory location
        // to hold the exception_ptr
        std::exception_ptr temp_excptr;
        auto excptr = exceptionPtr();
        if constexpr (std::is_same_v<typename T::Context, StreamContext>) {
          m_worker->emitPostModuleStreamPrefetchingSignal();
        } else if constexpr (std::is_same_v<typename T::Context, GlobalContext>) {
          m_worker->emitPostModuleGlobalPrefetchingSignal();
        }

        if (not excptr) {
          if (auto queue = m_worker->serializeRunModule()) {
            auto f = [worker = m_worker,
                      info = m_transitionInfo,
                      streamID = m_streamID,
                      parentContext = m_parentContext,
                      sContext = m_context,
                      serviceToken = m_serviceToken]() {
              //Need to make the services available
              ServiceRegistry::Operate operateRunModule(serviceToken.lock());

              //If needed, we pause the queue in begin transition and resume it
              // at the end transition. This can guarantee that the module
              // only processes one run or lumi at a time
              EnableQueueGuard enableQueueGuard{workerhelper::CallImpl<T>::enableGlobalQueue(worker)};
              std::exception_ptr ptr;
              worker->template runModuleAfterAsyncPrefetch<T>(ptr, info, streamID, parentContext, sContext);
            };
            //keep another global transition from running if necessary
            auto gQueue = workerhelper::CallImpl<T>::pauseGlobalQueue(m_worker);
            if (gQueue) {
              gQueue->push(*m_group, [queue, gQueue, f, group = m_group]() mutable {
                gQueue->pause();
                queue.push(*group, std::move(f));
              });
            } else {
              queue.push(*m_group, std::move(f));
            }
            return;
          }
        }

        m_worker->runModuleAfterAsyncPrefetch<T>(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
      }

    private:
      TransitionWorker<TI, TP>* m_worker;
      typename T::TransitionInfoType m_transitionInfo;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      typename T::Context const* m_context;
      ServiceWeakToken m_serviceToken;
      oneapi::tbb::task_group* m_group;
    };
  };
  namespace workerhelper {
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, TransitionActionGlobalBegin>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionGlobalBegin> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        // If preModuleSignal() throws, implDoBegin() is not called, and the
        // cpp destructor calls postModuleSignal (ignoring additional exceptions)
        cpp.preModuleSignal();
        // If implDoBegin() throws, the cpp destructor calls postModuleSignal
        // (ignoring additional exceptions)
        auto returnValue = iWorker->implDoBegin(info, mcc);
        // If postModuleSignal() throws, the exception will propagate to the framework
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsGlobalRuns();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return iWorker->globalRunsQueue();
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T*) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, TransitionActionStreamBegin>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionStreamBegin> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsStreamRuns();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T*) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, TransitionActionGlobalEnd>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionGlobalEnd> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        bool returnValue = true;
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          returnValue = iWorker->implDoEnd(info, mcc);
          cpp.postModuleSignal();
        }
        //The existence of noRunLumiSort option can shouldWriteRun() to retur kNo.
        if (iWorker->wantsWrites() and info.principal().shouldWriteRun() != edm::RunPrincipal::ShouldWriteRun::kNo) {
          auto sentry = signalslot::make_sentry(
              [actReg, context, mcc]() { actReg->postModuleWriteRunSignal_.emit(*context, *mcc); });
          actReg->preModuleWriteRunSignal_.emit(*context, *mcc);
          returnValue = iWorker->implDoWrite(info, mcc);
          sentry.succeeded();
        }
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsGlobalRuns() or iWorker->wantsWrites();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return iWorker->globalRunsQueue();
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, TransitionActionStreamEnd>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionStreamEnd> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
          cpp.postModuleSignal();
          return returnValue;
        }
        return true;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsStreamRuns();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalBegin>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalBegin>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoBegin(info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsGlobalLuminosityBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return iWorker->globalLuminosityBlocksQueue();
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamBegin>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamBegin>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsStreamLuminosityBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalEnd>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalEnd>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        bool returnValue = true;
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;
          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          returnValue = iWorker->implDoEnd(info, mcc);
          cpp.postModuleSignal();
        }
        //The existence of noRunLumiSort option can cause shouldWriteRun() to return kNo.
        if (iWorker->wantsWrites() and
            info.principal().shouldWriteLumi() != edm::LuminosityBlockPrincipal::ShouldWriteLumi::kNo) {
          auto sentry = signalslot::make_sentry(
              [actReg, context, &mcc]() { actReg->postModuleWriteLumiSignal_.emit(*context, *mcc); });
          actReg->preModuleWriteLumiSignal_.emit(*context, *mcc);
          iWorker->implDoWrite(info, mcc);
          sentry.succeeded();
        }
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsGlobalLuminosityBlocks() or iWorker->wantsWrites();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return iWorker->globalLuminosityBlocksQueue();
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamEnd>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamEnd>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
          cpp.postModuleSignal();
          return returnValue;
        }
        return true;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsStreamLuminosityBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalBegin>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalBegin>;
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoBeginProcessBlock(info.principal(), mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  ProcessBlockTransitionInfo const&,
                                  Transition) noexcept {}
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsProcessBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, TransitionActionProcessBlockInput>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, TransitionActionProcessBlockInput>;
      static bool call(TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       InputProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoAccessInputProcessBlock(info.principal(), mcc);
        cpp.postModuleSignal();
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  InputProcessBlockTransitionInfo const&,
                                  Transition) noexcept {}
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsInputProcessBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalEnd>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalEnd>;
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          auto returnValue = iWorker->implDoEndProcessBlock(info.principal(), mcc);
          cpp.postModuleSignal();
          return returnValue;
        }
        return true;
      }
      template <typename T>
      static void esPrefetchAsync(
          T*, WaitingTaskHolder, ServiceToken const&, ProcessBlockTransitionInfo const&, Transition) noexcept {}
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return iWorker->wantsProcessBlocks();
      }
      template <typename T>
      static bool needToRunSelection(T const* iWorker) noexcept {
        return false;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
  }  // namespace workerhelper

  template <typename TI, typename TP>
  template <typename T>
  void TransitionWorker<TI, TP>::prefetchAsync(WaitingTaskHolder iTask,
                                               ServiceToken const& token,
                                               ParentContext const& parentContext,
                                               typename T::TransitionInfoType const& transitionInfo,
                                               Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    if constexpr (std::is_same_v<typename T::Context, StreamContext>) {
      actReg_->preModuleStreamPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    } else if constexpr (std::is_same_v<typename T::Context, GlobalContext>) {
      actReg_->preModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);
    }

    workerhelper::CallImpl<T>::esPrefetchAsync(this, iTask, token, transitionInfo, iTransition);
    edPrefetchAsync(iTask, token, principal);
  }

  template <typename TI, typename TP>
  template <typename T>
  void TransitionWorker<TI, TP>::doWorkAsyncImpl(WaitingTaskHolder task,
                                                 typename T::TransitionInfoType const& transitionInfo,
                                                 ServiceToken const& token,
                                                 StreamID streamID,
                                                 ParentContext const& parentContext,
                                                 typename T::Context const* context) noexcept {
    if (not workerhelper::CallImpl<T>::wantsTransition(this)) {
      return;
    }

    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      WaitingTask* moduleTask =
          new RunModuleTask<T>(this, transitionInfo, token, streamID, parentContext, context, task.group());
      auto group = task.group();
      prefetchAsync<T>(WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, T::transition_);
    }
  }

  template <typename TI, typename TP>
  template <typename T>
  std::exception_ptr TransitionWorker<TI, TP>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      typename T::TransitionInfoType const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      typename T::Context const* context) noexcept {
    std::exception_ptr exceptionPtr;
    bool shouldRun = true;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, T::isEvent_, shouldTryToContinue_)) {
        exceptionPtr = iEPtr;
        setException(exceptionPtr);
        shouldRun = false;
      } else {
        if (not shouldTryToContinue_) {
          setPassed();
          shouldRun = false;
        }
      }
    }
    if (shouldRun) {
      // Caught exception is propagated via WaitingTaskList
      CMS_SA_ALLOW try { runModule<T>(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  template <typename TI, typename TP>
  template <typename T>
  void TransitionWorker<TI, TP>::doWorkNoPrefetchingAsyncImpl(WaitingTaskHolder task,
                                                              typename T::TransitionInfoType const& transitionInfo,
                                                              ServiceToken const& serviceToken,
                                                              StreamID streamID,
                                                              ParentContext const& parentContext,
                                                              typename T::Context const* context) noexcept {
    if (not workerhelper::CallImpl<T>::wantsTransition(this)) {
      return;
    }

    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    auto workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);
    if (workStarted) {
      ServiceWeakToken weakToken = serviceToken;
      auto toDo = [this, info = transitionInfo, streamID, parentContext, context, weakToken]() {
        std::exception_ptr exceptionPtr;
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          //Need to make the services available
          ServiceRegistry::Operate guard(weakToken.lock());

          this->runModule<T>(info, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };

      if (needsESPrefetching(T::transition_)) {
        auto group = task.group();
        auto afterPrefetch =
            edm::make_waiting_task([toDo = std::move(toDo), group, this](std::exception_ptr const* iExcept) {
              if (iExcept) {
                this->waitingTasks_.doneWaiting(*iExcept);
              } else {
                if (auto queue = this->serializeRunModule()) {
                  queue.push(*group, toDo);
                } else {
                  group->run(toDo);
                }
              }
            });
        moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);
        esPrefetchAsync(
            WaitingTaskHolder(*group, afterPrefetch), transitionInfo.eventSetupImpl(), T::transition_, serviceToken);
      } else {
        auto group = task.group();
        if (auto queue = this->serializeRunModule()) {
          queue.push(*group, toDo);
        } else {
          group->run(toDo);
        }
      }
    }
  }

  template <typename TI, typename TP>
  template <typename T>
  bool TransitionWorker<TI, TP>::runModule(typename T::TransitionInfoType const& transitionInfo,
                                           StreamID streamID,
                                           ParentContext const& parentContext,
                                           typename T::Context const* context) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallImpl<T>::call(
            this, streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed();
        } else {
          setFailed();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, T::isEvent_, shouldTryToContinue_)) {
        assert(not cached_exception_);
        setException(std::current_exception());
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed();
      }
    }

    return rc;
  }

}  // namespace edm
#endif
