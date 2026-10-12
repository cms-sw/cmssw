#ifndef FWCore_Framework_TransitionWorker_Event_h
#define FWCore_Framework_TransitionWorker_Event_h

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
#include "FWCore/Framework/interface/TransitionEdge.h"
#include "FWCore/Framework/interface/maker/TransitionWorker_Common.h"
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

  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  template <>
  class TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal> : public Worker {
  public:
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
        : Worker(iMD, iActions),
          numberOfPathsOn_(0),
          numberOfPathsLeftToRun_(0),
          earlyDeleteHelper_(nullptr),
          ranAcquireWithoutException_(false) {}

    virtual bool hasAccumulator() const noexcept = 0;

    //Run only for OutputModules. Causes TriggerResults information to be used to decide if module should run.
    void prePrefetchSelectionAsync(oneapi::tbb::task_group&,
                                   WaitingTask* task,
                                   ServiceToken const&,
                                   StreamID stream,
                                   EventPrincipal const*) noexcept;

    void prePrefetchSelectionAsync(
        oneapi::tbb::task_group&, WaitingTask* task, ServiceToken const&, StreamID stream, void const*) noexcept {
      assert(false);
    }

    //Called by Path to inject PathStatus and StreamSchedule to inject TriggerResults and PathStatus for empty paths.
    std::exception_ptr runModuleDirectly(EventTransitionInfo const&,
                                         StreamID,
                                         ParentContext const&,
                                         StreamContext const*) noexcept;

    //called by TransformingProductResolver so only for global Event
    virtual size_t transformIndex(edm::ProductDescription const&) const noexcept = 0;
    void doTransformAsync(WaitingTaskHolder,
                          size_t iTransformIndex,
                          EventPrincipal const&,
                          ServiceToken const&,
                          StreamID,
                          ModuleCallingContext const&,
                          StreamContext const*) noexcept;

    // Called if filter earlier in the path has failed.
    void skipOnPath(EventPrincipal const& iEvent);

    void reset() {
      resetBase();
      numberOfPathsLeftToRun_ = numberOfPathsOn_;
    }

    void postDoEvent(EventPrincipal const&);

    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);

    //Only for global Event
    void addedToPath() noexcept { ++numberOfPathsOn_; }

    template <TransitionEdge E>
    void doWorkAsync(WaitingTaskHolder iTask,
                     EventTransitionInfo const& iTransitionInfo,
                     ServiceToken const& iToken,
                     StreamID iStreamID,
                     ParentContext const& iParentContext,
                     StreamContext const* iContext) noexcept {
      if constexpr (E == TransitionEdge::kBegin) {
        this->doWorkAsyncImpl(std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
      }
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoPrefetchingAsync(WaitingTaskHolder iTask,
                                  EventTransitionInfo const& iTransitionInfo,
                                  ServiceToken const& iToken,
                                  StreamID iStreamID,
                                  ParentContext const& iParentContext,
                                  StreamContext const* iContext) noexcept {
      if constexpr (E == TransitionEdge::kBegin) {
        this->doWorkNoPrefetchingAsyncImpl(
            std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
      }
    }

  private:
    //Called by GlobalSchedule::processOneGlobalAsync, UnscheduledCallProducer::runAccumulatorsAsync, WorkerInPath::runWorkerAsync, UnscheduledProductResolver::prefetchAsync_
    void doWorkAsyncImpl(WaitingTaskHolder,
                         EventTransitionInfo const&,
                         ServiceToken const&,
                         StreamID,
                         ParentContext const&,
                         StreamContext const*) noexcept;

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    void doWorkNoPrefetchingAsyncImpl(WaitingTaskHolder,
                                      EventTransitionInfo const&,
                                      ServiceToken const&,
                                      StreamID,
                                      ParentContext const&,
                                      StreamContext const*) noexcept;

  protected:
    virtual bool implDo(EventTransitionInfo const&, ModuleCallingContext const*) = 0;

    virtual void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual bool implNeedToRunSelection() const noexcept = 0;

    virtual void implDoAcquire(EventTransitionInfo const&, ModuleCallingContext const*, WaitingTaskHolder&&) = 0;

    virtual void implDoTransformAsync(WaitingTaskHolder,
                                      size_t iTransformIndex,
                                      EventPrincipal const&,
                                      ParentContext const&,
                                      ServiceWeakToken const&) noexcept = 0;
    virtual ProductResolverIndex itemToGetForTransform(size_t iTransformIndex) const noexcept = 0;

    virtual bool implDoPrePrefetchSelection(StreamID, EventPrincipal const&, ModuleCallingContext const*) = 0;

  private:
    bool call(StreamID,
              EventTransitionInfo const& info,
              ActivityRegistry* actReg,
              ModuleCallingContext* mcc,
              StreamContext const* context) {
      using OccTraits = OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin>;

      //Want postDoEvent to be called after signals are sent.
      auto postSentry = make_sentry(this, [&](auto* worker) { worker->postDoEvent(info.principal()); });
      ModuleSignalSentry<OccTraits> signalSentry(actReg, context, mcc);
      signalSentry.preModuleSignal();
      bool returnValue;
      {
        ModuleCallingContextSentry mccSentry(*mcc);
        returnValue = this->implDo(info, mcc);
        mccSentry.finished(returnValue);
      }
      signalSentry.postModuleSignal();
      return returnValue;
    }
    bool needToRunSelection() const noexcept { return this->implNeedToRunSelection(); }

    bool runModule(EventTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);

    //Only used by Event and only for OutputModule
    virtual void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                              ModuleCallingContext const& moduleCallingContext,
                                              Principal const& iPrincipal) const noexcept = 0;

    void prefetchAsync(
        WaitingTaskHolder, ServiceToken const&, ParentContext const&, EventTransitionInfo const&, Transition) noexcept;

    bool needsESPrefetching(Transition iTrans) const noexcept {
      return iTrans < edm::Transition::NumberOfEventSetupTransitions ? not esItemsToGetFrom(iTrans).empty() : false;
    }

    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    virtual bool hasAcquire() const noexcept = 0;

    std::exception_ptr runModuleAfterAsyncPrefetch(
        std::exception_ptr, EventTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;

    // runAcquire() must take a copy of WaitingTaskHolder
    // see comment in runAcquireAfterAsyncPrefetch() definition
    void runAcquire(EventTransitionInfo const&, ParentContext const&, WaitingTaskHolder);

    //Only for Event
    void runAcquireAfterAsyncPrefetch(std::exception_ptr,
                                      EventTransitionInfo const&,
                                      ParentContext const&,
                                      WaitingTaskHolder) noexcept;

    std::exception_ptr handleExternalWorkException(std::exception_ptr iEPtr,
                                                   ParentContext const& parentContext) noexcept;

    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                    EventTransitionInfo const& transitionInfo,
                    ServiceToken const& token,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    StreamContext const* context,
                    oneapi::tbb::task_group* iGroup) noexcept
          : m_worker(worker),
            m_transitionInfo(transitionInfo),
            m_streamID(streamID),
            m_parentContext(parentContext),
            m_context(context),
            m_serviceToken(token),
            m_group(iGroup) {}

      void execute() final {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken.lock());

        using OccTraits = OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin>;
        //incase the emit causes an exception, we need a memory location
        // to hold the exception_ptr
        std::exception_ptr temp_excptr;
        auto excptr = exceptionPtr();
        if (!m_worker->hasAcquire()) {
          // Caught exception is passed to Worker::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskList
          CMS_SA_ALLOW try {
            //pre was called in prefetchAsync
            m_worker->emitPostModuleEventPrefetchingSignal();
          } catch (...) {
            temp_excptr = std::current_exception();
            if (not excptr) {
              excptr = temp_excptr;
            }
          }
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

              std::exception_ptr ptr;
              worker->runModuleAfterAsyncPrefetch(ptr, info, streamID, parentContext, sContext);
            };
            queue.push(*m_group, std::move(f));
            return;
          }
        }

        m_worker->runModuleAfterAsyncPrefetch(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
      }

    private:
      TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
      EventTransitionInfo m_transitionInfo;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      StreamContext const* m_context;
      ServiceWeakToken m_serviceToken;
      oneapi::tbb::task_group* m_group;
    };

    class AcquireTask : public WaitingTask {
    public:
      AcquireTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                  EventTransitionInfo const& eventTransitionInfo,
                  ServiceToken const& token,
                  ParentContext const& parentContext,
                  WaitingTaskHolder holder) noexcept
          : m_worker(worker),
            m_eventTransitionInfo(eventTransitionInfo),
            m_parentContext(parentContext),
            m_holder(std::move(holder)),
            m_serviceToken(token) {}

      void execute() final {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken.lock());

        //incase the emit causes an exception, we need a memory location
        // to hold the exception_ptr
        std::exception_ptr temp_excptr;
        auto excptr = exceptionPtr();
        // Caught exception is passed to Worker::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskHolder
        CMS_SA_ALLOW try {
          //pre was called in prefetchAsync
          m_worker->emitPostModuleEventPrefetchingSignal();
        } catch (...) {
          temp_excptr = std::current_exception();
          if (not excptr) {
            excptr = temp_excptr;
          }
        }

        if (not excptr) {
          if (auto queue = m_worker->serializeRunModule()) {
            queue.push(*m_holder.group(),
                       [worker = m_worker,
                        info = m_eventTransitionInfo,
                        parentContext = m_parentContext,
                        serviceToken = m_serviceToken,
                        holder = std::move(m_holder)]() mutable {
                         //Need to make the services available
                         ServiceRegistry::Operate operateRunAcquire(serviceToken.lock());

                         std::exception_ptr ptr;
                         worker->runAcquireAfterAsyncPrefetch(ptr, info, parentContext, std::move(holder));
                       });
            return;
          }
        }

        m_worker->runAcquireAfterAsyncPrefetch(excptr, m_eventTransitionInfo, m_parentContext, std::move(m_holder));
      }

    private:
      TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
      EventTransitionInfo m_eventTransitionInfo;
      ParentContext const m_parentContext;
      WaitingTaskHolder m_holder;
      ServiceWeakToken m_serviceToken;
    };

    // This class does nothing unless there is an exception originating
    // in an "External Worker". In that case, it handles converting the
    // exception to a CMS exception and adding context to the exception.
    class HandleExternalWorkExceptionTask : public WaitingTask {
    public:
      HandleExternalWorkExceptionTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                                      oneapi::tbb::task_group* group,
                                      WaitingTask* runModuleTask,
                                      ParentContext const& parentContext) noexcept;

      void execute() final;

    private:
      TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
      WaitingTask* m_runModuleTask;
      oneapi::tbb::task_group* m_group;
      ParentContext const m_parentContext;
    };

    int numberOfPathsOn_;
    std::atomic<int> numberOfPathsLeftToRun_;

    edm::propagate_const<EarlyDeleteHelper*> earlyDeleteHelper_;

    bool ranAcquireWithoutException_;
  };

  inline void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::prefetchAsync(
      WaitingTaskHolder iTask,
      ServiceToken const& token,
      ParentContext const& parentContext,
      EventTransitionInfo const& transitionInfo,
      Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    actReg_->preModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);

    esPrefetchAsync(iTask, transitionInfo.eventSetupImpl(), iTransition, token);
    edPrefetchAsync(iTask, token, principal);

    preActionBeforeRunEventAsync(iTask, moduleCallingContext_, principal);
  }

  inline void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl(
      WaitingTaskHolder task,
      EventTransitionInfo const& transitionInfo,
      ServiceToken const& token,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    using OccTraits = OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin>;

    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      //if have TriggerResults based selection we want to reject the event before doing prefetching
      if (needToRunSelection()) {
        //We need to run the selection in a different task so that
        // we can prefetch the data needed for the selection
        WaitingTask* moduleTask =
            new RunModuleTask(this, transitionInfo, token, streamID, parentContext, context, task.group());

        //make sure the task is either run or destroyed
        struct DestroyTask {
          DestroyTask(edm::WaitingTask* iTask) noexcept : m_task(iTask) {}

          ~DestroyTask() noexcept {
            auto p = m_task.exchange(nullptr);
            if (p) {
              TaskSentry s{p};
            }
          }

          edm::WaitingTask* release() noexcept { return m_task.exchange(nullptr); }

        private:
          std::atomic<edm::WaitingTask*> m_task;
        };
        if (hasAcquire()) {
          auto ownRunTask = std::make_shared<DestroyTask>(moduleTask);
          ServiceWeakToken weakToken = token;
          auto* group = task.group();
          moduleTask = make_waiting_task(
              [this, weakToken, transitionInfo, parentContext, ownRunTask, group](std::exception_ptr const* iExcept) {
                WaitingTaskHolder runTaskHolder(
                    *group, new HandleExternalWorkExceptionTask(this, group, ownRunTask->release(), parentContext));
                AcquireTask t(this, transitionInfo, weakToken.lock(), parentContext, runTaskHolder);
                t.execute();
              });
        }
        auto* group = task.group();
        auto ownModuleTask = std::make_shared<DestroyTask>(moduleTask);
        ServiceWeakToken weakToken = token;
        auto selectionTask =
            make_waiting_task([ownModuleTask, parentContext, info = transitionInfo, weakToken, group, this](
                                  std::exception_ptr const*) mutable {
              ServiceRegistry::Operate guard(weakToken.lock());
              prefetchAsync(WaitingTaskHolder(*group, ownModuleTask->release()),
                            weakToken.lock(),
                            parentContext,
                            info,
                            OccTraits::transition_);
            });
        prePrefetchSelectionAsync(*group, selectionTask, token, streamID, &transitionInfo.principal());
      } else {
        WaitingTask* moduleTask =
            new RunModuleTask(this, transitionInfo, token, streamID, parentContext, context, task.group());
        auto group = task.group();
        if (hasAcquire()) {
          WaitingTaskHolder runTaskHolder(*group,
                                          new HandleExternalWorkExceptionTask(this, group, moduleTask, parentContext));
          moduleTask = new AcquireTask(this, transitionInfo, token, parentContext, std::move(runTaskHolder));
        }
        prefetchAsync(
            WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, OccTraits::transition_);
      }
    }
  }

  inline std::exception_ptr TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      EventTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    std::exception_ptr exceptionPtr;
    bool shouldRun = true;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, true, shouldTryToContinue_)) {
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
      CMS_SA_ALLOW try { runModule(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  inline void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::doWorkNoPrefetchingAsyncImpl(
      WaitingTaskHolder task,
      EventTransitionInfo const& transitionInfo,
      ServiceToken const& serviceToken,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    using OccTraits = OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin>;

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
          this->runModule(info, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };

      if (needsESPrefetching(OccTraits::transition_)) {
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
                        OccTraits::transition_,
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

  inline bool TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModule(
      EventTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) {
    //unscheduled producers should advance this
    //if (T::isEvent_) {
    //  ++timesVisited_;
    //}
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = call(streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed();
        } else {
          setFailed();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, true, shouldTryToContinue_)) {
        assert(not cached_exception_);
        setException(std::current_exception());
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed();
      }
    }

    return rc;
  }

  inline std::exception_ptr TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModuleDirectly(
      EventTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    std::exception_ptr prefetchingException;  // null because there was no prefetching to do
    return runModuleAfterAsyncPrefetch(prefetchingException, transitionInfo, streamID, parentContext, context);
  }
}  // namespace edm
#endif
