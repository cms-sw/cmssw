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
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
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
  class EventSetupImpl;
  class ProductResolverIndexAndSkipBit;

  namespace workerhelper {
    template <typename TI, TransitionEdge E>
    class CallGlobalImpl;
  }
  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  class RunTransitionInfo;
  class LumiTransitionInfo;
  struct TransitionPhaseGlobal;

  template <typename TI, TransitionEdge E>
  struct TransitionTrait;
  template <>
  struct TransitionTrait<RunTransitionInfo, TransitionEdge::kBegin> {
    static constexpr Transition value = Transition::BeginRun;
  };
  template <>
  struct TransitionTrait<RunTransitionInfo, TransitionEdge::kEnd> {
    static constexpr Transition value = Transition::EndRun;
  };
  template <>
  struct TransitionTrait<LumiTransitionInfo, TransitionEdge::kBegin> {
    static constexpr Transition value = Transition::BeginLuminosityBlock;
  };
  template <>
  struct TransitionTrait<LumiTransitionInfo, TransitionEdge::kEnd> {
    static constexpr Transition value = Transition::EndLuminosityBlock;
  };

  template <typename TI, typename TP>
  class TransitionWorker : public Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}

    virtual bool wantsGlobalTransitions() const noexcept = 0;
    virtual bool wantsWrites() const noexcept = 0;

    //returns non-nullptr if the module can only process one Transition at a time
    virtual SerialTaskQueue* globalTransitionsQueue() = 0;

    void reset() { resetBase(); }

    template <TransitionEdge E>
    void doWorkAsync(WaitingTaskHolder iTask,
                     TI const& iTransitionInfo,
                     ServiceToken const& iToken,
                     StreamID iStreamID,
                     ParentContext const& iParentContext,
                     GlobalContext const* iContext) noexcept {
      this->template doWorkAsyncImpl<E>(std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoEDPrefetchingAsync(WaitingTaskHolder iTask,
                                    TI const& iTransitionInfo,
                                    ServiceToken const& iToken,
                                    StreamID iStreamID,
                                    ParentContext const& iParentContext,
                                    GlobalContext const* iContext) noexcept {
      this->template doWorkNoEDPrefetchingAsyncImpl<E>(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

  protected:
    //Called by GlobalSchedule::processOneGlobalAsync, UnscheduledCallProducer::runAccumulatorsAsync, WorkerInPath::runWorkerAsync, UnscheduledProductResolver::prefetchAsync_
    template <TransitionEdge E>
    void doWorkAsyncImpl(WaitingTaskHolder,
                         TI const&,
                         ServiceToken const&,
                         StreamID,
                         ParentContext const&,
                         GlobalContext const*) noexcept;

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoEDPrefetchingAsyncImpl(WaitingTaskHolder,
                                        TI const&,
                                        ServiceToken const&,
                                        StreamID,
                                        ParentContext const&,
                                        GlobalContext const*) noexcept;

    template <typename TINFO, TransitionEdge E>
    friend class workerhelper::CallGlobalImpl;

    virtual bool implDoBegin(TI const&, ModuleCallingContext const*) = 0;
    virtual bool implDoEnd(TI const&, ModuleCallingContext const*) = 0;
    virtual bool implDoWrite(TI const&, ModuleCallingContext const*) = 0;

  private:
    template <TransitionEdge E>
    bool runModule(TI const&, StreamID, ParentContext const&, GlobalContext const*);

    void prefetchAsync(WaitingTaskHolder, ServiceToken const&, ParentContext const&, TI const&, Transition) noexcept;

    void emitPostModuleGlobalPrefetchingSignal() {
      actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(),
                                                       moduleCallingContext_);
    }

    template <TransitionEdge E>
    std::exception_ptr runModuleAfterAsyncPrefetch(
        std::exception_ptr, TI const&, StreamID, ParentContext const&, GlobalContext const*) noexcept;

    template <TransitionEdge E>
    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(TransitionWorker<TI, TP>* worker,
                    TI const& transitionInfo,
                    ServiceToken const& token,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    GlobalContext const* context,
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
        m_worker->emitPostModuleGlobalPrefetchingSignal();

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
              SerialTaskQueue* gQueue = nullptr;
              if constexpr (E == TransitionEdge::kEnd) {
                gQueue = worker->globalTransitionsQueue();
              }
              EnableQueueGuard enableQueueGuard{gQueue};
              std::exception_ptr ptr;
              worker->template runModuleAfterAsyncPrefetch<E>(ptr, info, streamID, parentContext, sContext);
            };
            //keep another global transition from running if necessary
            SerialTaskQueue* gQueue = nullptr;
            if constexpr (E == TransitionEdge::kBegin) {
              gQueue = m_worker->globalTransitionsQueue();
            }
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

        m_worker->runModuleAfterAsyncPrefetch<E>(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
      }

    private:
      TransitionWorker<TI, TP>* m_worker;
      TI m_transitionInfo;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      GlobalContext const* m_context;
      ServiceWeakToken m_serviceToken;
      oneapi::tbb::task_group* m_group;
    };
  };
  namespace workerhelper {
    template <>
    class CallGlobalImpl<RunTransitionInfo, TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context) {
        struct SignalTrait {
          using Context = GlobalContext;
          static void preModuleSignal(ActivityRegistry* a,
                                      GlobalContext const* globalContext,
                                      ModuleCallingContext const* moduleCallingContext) {
            a->preModuleGlobalBeginRunSignal_.emit(*globalContext, *moduleCallingContext);
          }
          static void postModuleSignal(ActivityRegistry* a,
                                       GlobalContext const* globalContext,
                                       ModuleCallingContext const* moduleCallingContext) {
            a->postModuleGlobalBeginRunSignal_.emit(*globalContext, *moduleCallingContext);
          }
        };
        ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
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
    };
    template <>
    class CallGlobalImpl<RunTransitionInfo, TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context) {
        bool returnValue = true;
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          struct SignalTrait {
            using Context = GlobalContext;

            static void preModuleSignal(ActivityRegistry* a,
                                        GlobalContext const* globalContext,
                                        ModuleCallingContext const* moduleCallingContext) {
              a->preModuleGlobalEndRunSignal_.emit(*globalContext, *moduleCallingContext);
            }
            static void postModuleSignal(ActivityRegistry* a,
                                         GlobalContext const* globalContext,
                                         ModuleCallingContext const* moduleCallingContext) {
              a->postModuleGlobalEndRunSignal_.emit(*globalContext, *moduleCallingContext);
            }
          };
          ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
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
    };
    template <>
    class CallGlobalImpl<LumiTransitionInfo, TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context) {
        struct SignalTrait {
          using Context = GlobalContext;
          static void preModuleSignal(ActivityRegistry* a,
                                      GlobalContext const* globalContext,
                                      ModuleCallingContext const* moduleCallingContext) {
            a->preModuleGlobalBeginLumiSignal_.emit(*globalContext, *moduleCallingContext);
          }
          static void postModuleSignal(ActivityRegistry* a,
                                       GlobalContext const* globalContext,
                                       ModuleCallingContext const* moduleCallingContext) {
            a->postModuleGlobalBeginLumiSignal_.emit(*globalContext, *moduleCallingContext);
          }
        };
        ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoBegin(info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
    };
    template <>
    class CallGlobalImpl<LumiTransitionInfo, TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context) {
        bool returnValue = true;
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;
          struct SignalTrait {
            using Context = GlobalContext;
            static void preModuleSignal(ActivityRegistry* a,
                                        GlobalContext const* globalContext,
                                        ModuleCallingContext const* moduleCallingContext) {
              a->preModuleGlobalEndLumiSignal_.emit(*globalContext, *moduleCallingContext);
            }
            static void postModuleSignal(ActivityRegistry* a,
                                         GlobalContext const* globalContext,
                                         ModuleCallingContext const* moduleCallingContext) {
              a->postModuleGlobalEndLumiSignal_.emit(*globalContext, *moduleCallingContext);
            }
          };
          ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
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
    };
  }  // namespace workerhelper

  template <typename TI, typename TP>
  void TransitionWorker<TI, TP>::prefetchAsync(WaitingTaskHolder iTask,
                                               ServiceToken const& token,
                                               ParentContext const& parentContext,
                                               TI const& transitionInfo,
                                               Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    actReg_->preModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);

    esPrefetchAsync(iTask, transitionInfo.eventSetupImpl(), iTransition, token);

    edPrefetchAsync(iTask, token, principal);
  }

  template <typename TI, typename TP>
  template <TransitionEdge E>
  void TransitionWorker<TI, TP>::doWorkAsyncImpl(WaitingTaskHolder task,
                                                 TI const& transitionInfo,
                                                 ServiceToken const& token,
                                                 StreamID streamID,
                                                 ParentContext const& parentContext,
                                                 GlobalContext const* context) noexcept {
    if constexpr (E == TransitionEdge::kBegin) {
      if (not wantsGlobalTransitions()) {
        //This module wants a write without a global transition.
        return;
      }
    }

    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      WaitingTask* moduleTask =
          new RunModuleTask<E>(this, transitionInfo, token, streamID, parentContext, context, task.group());
      auto group = task.group();
      prefetchAsync(
          WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, TransitionTrait<TI, E>::value);
    }
  }

  template <typename TI, typename TP>
  template <TransitionEdge E>
  std::exception_ptr TransitionWorker<TI, TP>::runModuleAfterAsyncPrefetch(std::exception_ptr iEPtr,
                                                                           TI const& transitionInfo,
                                                                           StreamID streamID,
                                                                           ParentContext const& parentContext,
                                                                           GlobalContext const* context) noexcept {
    std::exception_ptr exceptionPtr;
    bool shouldRun = true;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, false, shouldTryToContinue_)) {
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
      CMS_SA_ALLOW try { runModule<E>(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  template <typename TI, typename TP>
  template <TransitionEdge E>
  void TransitionWorker<TI, TP>::doWorkNoEDPrefetchingAsyncImpl(WaitingTaskHolder task,
                                                                TI const& transitionInfo,
                                                                ServiceToken const& serviceToken,
                                                                StreamID streamID,
                                                                ParentContext const& parentContext,
                                                                GlobalContext const* context) noexcept {
    if constexpr (E == TransitionEdge::kBegin) {
      // This module wants a begin transition
      if (not wantsGlobalTransitions()) {
        return;
      }
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

          this->runModule<E>(info, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };

      if (needsESPrefetching(TransitionTrait<TI, E>::value)) {
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
        esPrefetchAsync(WaitingTaskHolder(*group, afterPrefetch),
                        transitionInfo.eventSetupImpl(),
                        TransitionTrait<TI, E>::value,
                        serviceToken);
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
  template <TransitionEdge E>
  bool TransitionWorker<TI, TP>::runModule(TI const& transitionInfo,
                                           StreamID streamID,
                                           ParentContext const& parentContext,
                                           GlobalContext const* context) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallGlobalImpl<TI, E>::call(
            this, streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed();
        } else {
          setFailed();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, false, shouldTryToContinue_)) {
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
