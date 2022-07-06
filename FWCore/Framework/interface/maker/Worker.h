#ifndef FWCore_Framework_Worker_h
#define FWCore_Framework_Worker_h

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
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
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
  class ModuleProcessName;
  class ProductResolverIndexHelper;
  class ProductResolverIndexAndSkipBit;
  class StreamID;
  class StreamContext;
  class ProductRegistry;
  class ThinnedAssociationsHelper;

  namespace workerhelper {
    template <typename O>
    class CallImpl;
  }
  namespace eventsetup {
    class ESRecordsToProxyIndices;
  }

  class Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    enum Types { kAnalyzer, kFilter, kProducer, kOutputModule };
    enum ConcurrencyTypes { kGlobal, kLimited, kOne, kStream, kLegacy };
    struct TaskQueueAdaptor {
      SerialTaskQueueChain* serial_ = nullptr;
      LimitedTaskQueue* limited_ = nullptr;

      TaskQueueAdaptor() = default;
      TaskQueueAdaptor(SerialTaskQueueChain* iChain) : serial_(iChain) {}
      TaskQueueAdaptor(LimitedTaskQueue* iLimited) : limited_(iLimited) {}

      operator bool() { return serial_ != nullptr or limited_ != nullptr; }

      template <class F>
      void push(oneapi::tbb::task_group& iG, F&& iF) {
        if (serial_) {
          serial_->push(iG, iF);
        } else {
          limited_->push(iG, iF);
        }
      }
    };

    Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions);
    virtual ~Worker();

    Worker(Worker const&) = delete;             // Disallow copying and moving
    Worker& operator=(Worker const&) = delete;  // Disallow copying and moving

    void clearModule() {
      moduleValid_ = false;
      doClearModule();
    }

    virtual bool wantsProcessBlocks() const = 0;
    virtual bool wantsInputProcessBlocks() const = 0;
    virtual bool wantsGlobalRuns() const = 0;
    virtual bool wantsGlobalLuminosityBlocks() const = 0;
    virtual bool wantsStreamRuns() const = 0;
    virtual bool wantsStreamLuminosityBlocks() const = 0;

    virtual SerialTaskQueue* globalRunsQueue() = 0;
    virtual SerialTaskQueue* globalLuminosityBlocksQueue() = 0;

    void prePrefetchSelectionAsync(
        oneapi::tbb::task_group&, WaitingTask* task, ServiceToken const&, StreamID stream, EventPrincipal const*);

    void prePrefetchSelectionAsync(
        oneapi::tbb::task_group&, WaitingTask* task, ServiceToken const&, StreamID stream, void const*) {
      assert(false);
    }

    template <typename T>
    void doWorkAsync(WaitingTaskHolder,
                     typename T::TransitionInfoType const&,
                     ServiceToken const&,
                     StreamID,
                     ParentContext const&,
                     typename T::Context const*);

    template <typename T>
    void doWorkNoPrefetchingAsync(WaitingTaskHolder,
                                  typename T::TransitionInfoType const&,
                                  ServiceToken const&,
                                  StreamID,
                                  ParentContext const&,
                                  typename T::Context const*);

    template <typename T>
    std::exception_ptr runModuleDirectly(typename T::TransitionInfoType const&,
                                         StreamID,
                                         ParentContext const&,
                                         typename T::Context const*);

    void callWhenDoneAsync(WaitingTaskHolder task) { waitingTasks_.add(std::move(task)); }
    void skipOnPath(EventPrincipal const& iEvent);
    void beginJob();
    void endJob();
    void beginStream(StreamID id, StreamContext& streamContext);
    void endStream(StreamID id, StreamContext& streamContext);
    void respondToOpenInputFile(FileBlock const& fb) { implRespondToOpenInputFile(fb); }
    void respondToCloseInputFile(FileBlock const& fb) { implRespondToCloseInputFile(fb); }
    void respondToCloseOutputFile() { implRespondToCloseOutputFile(); }
    void registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper);

    void reset() {
      cached_exception_ = std::exception_ptr();
      state_ = Ready;
      waitingTasks_.reset();
      workStarted_ = false;
      numberOfPathsLeftToRun_ = numberOfPathsOn_;
    }

    void postDoEvent(EventPrincipal const&);

    ModuleDescription const* description() const {
      if (moduleValid_) {
        return moduleCallingContext_.moduleDescription();
      }
      return nullptr;
    }
    ///The signals are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void setActivityRegistry(std::shared_ptr<ActivityRegistry> areg);

    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);

    //Used to make EDGetToken work
    virtual void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&) = 0;
    virtual void updateLookup(eventsetup::ESRecordsToProxyIndices const&) = 0;
    virtual void selectInputProcessBlocks(ProductRegistry const&, ProcessBlockHelperBase const&) = 0;
    virtual void resolvePutIndicies(
        BranchType iBranchType,
        std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
            iIndicies) = 0;

    virtual void modulesWhoseProductsAreConsumed(
        std::array<std::vector<ModuleDescription const*>*, NumBranchTypes>& modules,
        std::vector<ModuleProcessName>& modulesInPreviousProcesses,
        ProductRegistry const& preg,
        std::map<std::string, ModuleDescription const*> const& labelsToDesc) const = 0;

    virtual void convertCurrentProcessAlias(std::string const& processName) = 0;

    virtual std::vector<ConsumesInfo> consumesInfo() const = 0;

    virtual Types moduleType() const = 0;
    virtual ConcurrencyTypes moduleConcurrencyType() const = 0;

    void clearCounters() {
      timesRun_.store(0, std::memory_order_release);
      timesVisited_.store(0, std::memory_order_release);
      timesPassed_.store(0, std::memory_order_release);
      timesFailed_.store(0, std::memory_order_release);
      timesExcept_.store(0, std::memory_order_release);
    }

    void addedToPath() { ++numberOfPathsOn_; }
    //NOTE: calling state() is done to force synchronization across threads
    int timesRun() const { return timesRun_.load(std::memory_order_acquire); }
    int timesVisited() const { return timesVisited_.load(std::memory_order_acquire); }
    int timesPassed() const { return timesPassed_.load(std::memory_order_acquire); }
    int timesFailed() const { return timesFailed_.load(std::memory_order_acquire); }
    int timesExcept() const { return timesExcept_.load(std::memory_order_acquire); }
    State state() const { return state_; }

    int timesPass() const { return timesPassed(); }  // for backward compatibility only - to be removed soon

    virtual bool hasAccumulator() const = 0;

  protected:
    template <typename O>
    friend class workerhelper::CallImpl;

    virtual void doClearModule() = 0;

    virtual std::string workerType() const = 0;
    virtual bool implDo(EventTransitionInfo const&, ModuleCallingContext const*) = 0;

    virtual void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual bool implNeedToRunSelection() const = 0;

    virtual void implDoAcquire(EventTransitionInfo const&,
                               ModuleCallingContext const*,
                               WaitingTaskWithArenaHolder&) = 0;

    virtual bool implDoPrePrefetchSelection(StreamID, EventPrincipal const&, ModuleCallingContext const*) = 0;
    virtual bool implDoBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) = 0;
    virtual bool implDoAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) = 0;
    virtual bool implDoEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) = 0;
    virtual bool implDoBegin(RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamBegin(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoEnd(RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoBegin(LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamBegin(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoEnd(LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual void implBeginJob() = 0;
    virtual void implEndJob() = 0;
    virtual void implBeginStream(StreamID) = 0;
    virtual void implEndStream(StreamID) = 0;

    void resetModuleDescription(ModuleDescription const*);

    ActivityRegistry* activityRegistry() { return actReg_.get(); }

  private:
    template <typename T>
    bool runModule(typename T::TransitionInfoType const&, StreamID, ParentContext const&, typename T::Context const*);

    virtual void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;

    virtual std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType) const = 0;

    virtual std::vector<ESProxyIndex> const& esItemsToGetFrom(Transition) const = 0;
    virtual std::vector<ESRecordIndex> const& esRecordsToGetFrom(Transition) const = 0;
    virtual std::vector<ProductResolverIndex> const& itemsShouldPutInEvent() const = 0;

    virtual void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                              ModuleCallingContext const& moduleCallingContext,
                                              Principal const& iPrincipal) const = 0;

    virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseOutputFile() = 0;

    virtual void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) = 0;

    virtual TaskQueueAdaptor serializeRunModule() = 0;

    bool shouldRethrowException(std::exception_ptr iPtr, ParentContext const& parentContext, bool isEvent) const;

    template <bool IS_EVENT>
    bool setPassed() {
      if (IS_EVENT) {
        timesPassed_.fetch_add(1, std::memory_order_relaxed);
      }
      state_ = Pass;
      return true;
    }

    template <bool IS_EVENT>
    bool setFailed() {
      if (IS_EVENT) {
        timesFailed_.fetch_add(1, std::memory_order_relaxed);
      }
      state_ = Fail;
      return false;
    }

    template <bool IS_EVENT>
    std::exception_ptr setException(std::exception_ptr iException) {
      if (IS_EVENT) {
        timesExcept_.fetch_add(1, std::memory_order_relaxed);
      }
      cached_exception_ = iException;  // propagate_const<T> has no reset() function
      state_ = Exception;
      return cached_exception_;
    }

    template <typename T>
    void prefetchAsync(WaitingTaskHolder,
                       ServiceToken const&,
                       ParentContext const&,
                       typename T::TransitionInfoType const&,
                       Transition);

    void esPrefetchAsync(WaitingTaskHolder, EventSetupImpl const&, Transition, ServiceToken const&);
    void edPrefetchAsync(WaitingTaskHolder, ServiceToken const&, Principal const&) const;

    bool needsESPrefetching(Transition iTrans) const noexcept {
      return iTrans < edm::Transition::NumberOfEventSetupTransitions ? not esItemsToGetFrom(iTrans).empty() : false;
    }

    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    virtual bool hasAcquire() const = 0;

    template <typename T>
    std::exception_ptr runModuleAfterAsyncPrefetch(std::exception_ptr,
                                                   typename T::TransitionInfoType const&,
                                                   StreamID,
                                                   ParentContext const&,
                                                   typename T::Context const*);

    void runAcquire(EventTransitionInfo const&, ParentContext const&, WaitingTaskWithArenaHolder&);

    void runAcquireAfterAsyncPrefetch(std::exception_ptr,
                                      EventTransitionInfo const&,
                                      ParentContext const&,
                                      WaitingTaskWithArenaHolder);

    std::exception_ptr handleExternalWorkException(std::exception_ptr iEPtr, ParentContext const& parentContext);

    template <typename T>
    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(Worker* worker,
                    typename T::TransitionInfoType const& transitionInfo,
                    ServiceToken const& token,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    typename T::Context const* context,
                    oneapi::tbb::task_group* iGroup)
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
        if constexpr (T::isEvent_) {
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
      Worker* m_worker;
      typename T::TransitionInfoType m_transitionInfo;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      typename T::Context const* m_context;
      ServiceWeakToken m_serviceToken;
      oneapi::tbb::task_group* m_group;
    };

    // AcquireTask is only used for the Event case, but we define
    // it as a template so all cases will compile.
    // DUMMY exists to work around the C++ Standard prohibition on
    // fully specializing templates nested in other classes.
    template <typename T, typename DUMMY = void>
    class AcquireTask : public WaitingTask {
    public:
      AcquireTask(Worker*,
                  typename T::TransitionInfoType const&,
                  ServiceToken const&,
                  ParentContext const&,
                  WaitingTaskWithArenaHolder) {}
      void execute() final {}
    };

    template <typename DUMMY>
    class AcquireTask<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>, DUMMY> : public WaitingTask {
    public:
      AcquireTask(Worker* worker,
                  EventTransitionInfo const& eventTransitionInfo,
                  ServiceToken const& token,
                  ParentContext const& parentContext,
                  WaitingTaskWithArenaHolder holder)
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
        // Caught exception is passed to Worker::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskWithArenaHolder
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
                        holder = m_holder]() {
                         //Need to make the services available
                         ServiceRegistry::Operate operateRunAcquire(serviceToken.lock());

                         std::exception_ptr ptr;
                         worker->runAcquireAfterAsyncPrefetch(ptr, info, parentContext, holder);
                       });
            return;
          }
        }

        m_worker->runAcquireAfterAsyncPrefetch(excptr, m_eventTransitionInfo, m_parentContext, std::move(m_holder));
      }

    private:
      Worker* m_worker;
      EventTransitionInfo m_eventTransitionInfo;
      ParentContext const m_parentContext;
      WaitingTaskWithArenaHolder m_holder;
      ServiceWeakToken m_serviceToken;
    };

    // This class does nothing unless there is an exception originating
    // in an "External Worker". In that case, it handles converting the
    // exception to a CMS exception and adding context to the exception.
    class HandleExternalWorkExceptionTask : public WaitingTask {
    public:
      HandleExternalWorkExceptionTask(Worker* worker,
                                      oneapi::tbb::task_group* group,
                                      WaitingTask* runModuleTask,
                                      ParentContext const& parentContext);

      void execute() final;

    private:
      Worker* m_worker;
      WaitingTask* m_runModuleTask;
      oneapi::tbb::task_group* m_group;
      ParentContext const m_parentContext;
    };

    std::atomic<int> timesRun_;
    std::atomic<int> timesVisited_;
    std::atomic<int> timesPassed_;
    std::atomic<int> timesFailed_;
    std::atomic<int> timesExcept_;
    std::atomic<State> state_;
    int numberOfPathsOn_;
    std::atomic<int> numberOfPathsLeftToRun_;

    ModuleCallingContext moduleCallingContext_;

    ExceptionToActionTable const* actions_;                         // memory assumed to be managed elsewhere
    CMS_THREAD_GUARD(state_) std::exception_ptr cached_exception_;  // if state is 'exception'

    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.

    edm::propagate_const<EarlyDeleteHelper*> earlyDeleteHelper_;

    edm::WaitingTaskList waitingTasks_;
    std::atomic<bool> workStarted_;
    bool ranAcquireWithoutException_;
    bool moduleValid_ = true;
  };

  namespace {
    template <typename T>
    class ModuleSignalSentry {
    public:
      ModuleSignalSentry(ActivityRegistry* a,
                         typename T::Context const* context,
                         ModuleCallingContext const* moduleCallingContext)
          : a_(a), context_(context), moduleCallingContext_(moduleCallingContext) {
        if (a_)
          T::preModuleSignal(a_, context, moduleCallingContext_);
      }

      ~ModuleSignalSentry() {
        if (a_)
          T::postModuleSignal(a_, context_, moduleCallingContext_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      typename T::Context const* context_;
      ModuleCallingContext const* moduleCallingContext_;
    };

  }  // namespace

  namespace workerhelper {
    template <>
    class CallImpl<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>> {
    public:
      typedef OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker,
                       StreamID,
                       EventTransitionInfo const& info,
                       ActivityRegistry* /* actReg */,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* /* context*/) {
        //Signal sentry is handled by the module
        return iWorker->implDo(info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  EventTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return true; }
      static bool needToRunSelection(Worker const* iWorker) { return iWorker->implNeedToRunSelection(); }

      static SerialTaskQueue* pauseGlobalQueue(Worker*) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };

    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Arg;
      static bool call(Worker* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalRuns(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return iWorker->globalRunsQueue(); }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id, info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamRuns(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Arg;
      static bool call(Worker* iWorker,
                       StreamID,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalRuns(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker* iWorker) { return iWorker->globalRunsQueue(); }
    };
    template <>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>> {
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Arg;
      static bool call(Worker* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id, info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamRuns(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>;
      static bool call(Worker* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return iWorker->globalLuminosityBlocksQueue(); }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>;
      static bool call(Worker* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id, info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>;
      static bool call(Worker* iWorker,
                       StreamID,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker* iWorker) { return iWorker->globalLuminosityBlocksQueue(); }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;
      static bool call(Worker* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id, info, mcc);
      }
      static void esPrefetchAsync(Worker* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>;
      static bool call(Worker* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBeginProcessBlock(info.principal(), mcc);
      }
      static void esPrefetchAsync(
          Worker*, WaitingTaskHolder, ServiceToken const&, ProcessBlockTransitionInfo const&, Transition) {}
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsProcessBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>;
      static bool call(Worker* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoAccessInputProcessBlock(info.principal(), mcc);
      }
      static void esPrefetchAsync(
          Worker*, WaitingTaskHolder, ServiceToken const&, ProcessBlockTransitionInfo const&, Transition) {}
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsInputProcessBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>> {
    public:
      using Arg = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
      static bool call(Worker* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEndProcessBlock(info.principal(), mcc);
      }
      static void esPrefetchAsync(
          Worker*, WaitingTaskHolder, ServiceToken const&, ProcessBlockTransitionInfo const&, Transition) {}
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsProcessBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
  }  // namespace workerhelper

  template <typename T>
  void Worker::prefetchAsync(WaitingTaskHolder iTask,
                             ServiceToken const& token,
                             ParentContext const& parentContext,
                             typename T::TransitionInfoType const& transitionInfo,
                             Transition iTransition) {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    if (principal.branchType() == InEvent) {
      actReg_->preModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    workerhelper::CallImpl<T>::esPrefetchAsync(this, iTask, token, transitionInfo, iTransition);
    edPrefetchAsync(iTask, token, principal);

    if (principal.branchType() == InEvent) {
      preActionBeforeRunEventAsync(iTask, moduleCallingContext_, principal);
    }
  }

  template <typename T>
  void Worker::doWorkAsync(WaitingTaskHolder task,
                           typename T::TransitionInfoType const& transitionInfo,
                           ServiceToken const& token,
                           StreamID streamID,
                           ParentContext const& parentContext,
                           typename T::Context const* context) {
    if (not workerhelper::CallImpl<T>::wantsTransition(this)) {
      return;
    }

    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);
    if constexpr (T::isEvent_) {
      timesVisited_.fetch_add(1, std::memory_order_relaxed);
    }

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      //if have TriggerResults based selection we want to reject the event before doing prefetching
      if (workerhelper::CallImpl<T>::needToRunSelection(this)) {
        //We need to run the selection in a different task so that
        // we can prefetch the data needed for the selection
        WaitingTask* moduleTask =
            new RunModuleTask<T>(this, transitionInfo, token, streamID, parentContext, context, task.group());

        //make sure the task is either run or destroyed
        struct DestroyTask {
          DestroyTask(edm::WaitingTask* iTask) : m_task(iTask) {}

          ~DestroyTask() {
            auto p = m_task.exchange(nullptr);
            if (p) {
              TaskSentry s{p};
            }
          }

          edm::WaitingTask* release() { return m_task.exchange(nullptr); }

        private:
          std::atomic<edm::WaitingTask*> m_task;
        };
        if constexpr (T::isEvent_) {
          if (hasAcquire()) {
            auto ownRunTask = std::make_shared<DestroyTask>(moduleTask);
            ServiceWeakToken weakToken = token;
            auto* group = task.group();
            moduleTask = make_waiting_task(
                [this, weakToken, transitionInfo, parentContext, ownRunTask, group](std::exception_ptr const* iExcept) {
                  WaitingTaskWithArenaHolder runTaskHolder(
                      *group, new HandleExternalWorkExceptionTask(this, group, ownRunTask->release(), parentContext));
                  AcquireTask<T> t(this, transitionInfo, weakToken.lock(), parentContext, runTaskHolder);
                  t.execute();
                });
          }
        }
        auto* group = task.group();
        auto ownModuleTask = std::make_shared<DestroyTask>(moduleTask);
        ServiceWeakToken weakToken = token;
        auto selectionTask =
            make_waiting_task([ownModuleTask, parentContext, info = transitionInfo, weakToken, group, this](
                                  std::exception_ptr const*) mutable {
              ServiceRegistry::Operate guard(weakToken.lock());
              prefetchAsync<T>(WaitingTaskHolder(*group, ownModuleTask->release()),
                               weakToken.lock(),
                               parentContext,
                               info,
                               T::transition_);
            });
        prePrefetchSelectionAsync(*group, selectionTask, token, streamID, &transitionInfo.principal());
      } else {
        WaitingTask* moduleTask =
            new RunModuleTask<T>(this, transitionInfo, token, streamID, parentContext, context, task.group());
        auto group = task.group();
        if constexpr (T::isEvent_) {
          if (hasAcquire()) {
            WaitingTaskWithArenaHolder runTaskHolder(
                *group, new HandleExternalWorkExceptionTask(this, group, moduleTask, parentContext));
            moduleTask = new AcquireTask<T>(this, transitionInfo, token, parentContext, std::move(runTaskHolder));
          }
        }
        prefetchAsync<T>(WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, T::transition_);
      }
    }
  }

  template <typename T>
  std::exception_ptr Worker::runModuleAfterAsyncPrefetch(std::exception_ptr iEPtr,
                                                         typename T::TransitionInfoType const& transitionInfo,
                                                         StreamID streamID,
                                                         ParentContext const& parentContext,
                                                         typename T::Context const* context) {
    std::exception_ptr exceptionPtr;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, T::isEvent_)) {
        exceptionPtr = iEPtr;
        setException<T::isEvent_>(exceptionPtr);
      } else {
        setPassed<T::isEvent_>();
      }
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    } else {
      // Caught exception is propagated via WaitingTaskList
      CMS_SA_ALLOW try { runModule<T>(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  template <typename T>
  void Worker::doWorkNoPrefetchingAsync(WaitingTaskHolder task,
                                        typename T::TransitionInfoType const& transitionInfo,
                                        ServiceToken const& serviceToken,
                                        StreamID streamID,
                                        ParentContext const& parentContext,
                                        typename T::Context const* context) {
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

  template <typename T>
  bool Worker::runModule(typename T::TransitionInfoType const& transitionInfo,
                         StreamID streamID,
                         ParentContext const& parentContext,
                         typename T::Context const* context) {
    //unscheduled producers should advance this
    //if (T::isEvent_) {
    //  ++timesVisited_;
    //}
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    if constexpr (T::isEvent_) {
      timesRun_.fetch_add(1, std::memory_order_relaxed);
    }

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallImpl<T>::call(
            this, streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed<T::isEvent_>();
        } else {
          setFailed<T::isEvent_>();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, T::isEvent_)) {
        assert(not cached_exception_);
        setException<T::isEvent_>(std::current_exception());
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed<T::isEvent_>();
      }
    }

    return rc;
  }

  template <typename T>
  std::exception_ptr Worker::runModuleDirectly(typename T::TransitionInfoType const& transitionInfo,
                                               StreamID streamID,
                                               ParentContext const& parentContext,
                                               typename T::Context const* context) {
    timesVisited_.fetch_add(1, std::memory_order_relaxed);
    std::exception_ptr prefetchingException;  // null because there was no prefetching to do
    return runModuleAfterAsyncPrefetch<T>(prefetchingException, transitionInfo, streamID, parentContext, context);
  }
}  // namespace edm
#endif
