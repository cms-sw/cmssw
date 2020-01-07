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
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
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

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <atomic>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>

namespace edm {
  class EventPrincipal;
  class EarlyDeleteHelper;
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
    struct TaskQueueAdaptor {
      SerialTaskQueueChain* serial_ = nullptr;
      LimitedTaskQueue* limited_ = nullptr;

      TaskQueueAdaptor() = default;
      TaskQueueAdaptor(SerialTaskQueueChain* iChain) : serial_(iChain) {}
      TaskQueueAdaptor(LimitedTaskQueue* iLimited) : limited_(iLimited) {}

      operator bool() { return serial_ != nullptr or limited_ != nullptr; }

      template <class F>
      void push(F&& iF) {
        if (serial_) {
          serial_->push(iF);
        } else {
          limited_->push(iF);
        }
      }
      template <class F>
      void pushAndWait(F&& iF) {
        if (serial_) {
          serial_->pushAndWait(iF);
        } else {
          limited_->pushAndWait(iF);
        }
      }
    };

    Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions);
    virtual ~Worker();

    Worker(Worker const&) = delete;             // Disallow copying and moving
    Worker& operator=(Worker const&) = delete;  // Disallow copying and moving

    virtual bool wantsGlobalRuns() const = 0;
    virtual bool wantsGlobalLuminosityBlocks() const = 0;
    virtual bool wantsStreamRuns() const = 0;
    virtual bool wantsStreamLuminosityBlocks() const = 0;

    virtual SerialTaskQueue* globalRunsQueue() = 0;
    virtual SerialTaskQueue* globalLuminosityBlocksQueue() = 0;

    template <typename T>
    bool doWork(typename T::MyPrincipal const&,
                EventSetupImpl const& c,
                StreamID stream,
                ParentContext const& parentContext,
                typename T::Context const* context);

    void prePrefetchSelectionAsync(WaitingTask* task, ServiceToken const&, StreamID stream, EventPrincipal const*);

    void prePrefetchSelectionAsync(WaitingTask* task, ServiceToken const&, StreamID stream, void const*) {
      assert(false);
    }

    template <typename T>
    void doWorkAsync(WaitingTask* task,
                     typename T::MyPrincipal const&,
                     EventSetupImpl const& c,
                     ServiceToken const& token,
                     StreamID stream,
                     ParentContext const& parentContext,
                     typename T::Context const* context);

    template <typename T>
    void doWorkNoPrefetchingAsync(WaitingTask* task,
                                  typename T::MyPrincipal const&,
                                  EventSetupImpl const& c,
                                  ServiceToken const& token,
                                  StreamID stream,
                                  ParentContext const& parentContext,
                                  typename T::Context const* context);

    template <typename T>
    std::exception_ptr runModuleDirectly(typename T::MyPrincipal const& ep,
                                         EventSetupImpl const& es,
                                         StreamID streamID,
                                         ParentContext const& parentContext,
                                         typename T::Context const* context);

    void callWhenDoneAsync(WaitingTask* task) { waitingTasks_.add(task); }
    void skipOnPath();
    void beginJob();
    void endJob();
    void beginStream(StreamID id, StreamContext& streamContext);
    void endStream(StreamID id, StreamContext& streamContext);
    void respondToOpenInputFile(FileBlock const& fb) { implRespondToOpenInputFile(fb); }
    void respondToCloseInputFile(FileBlock const& fb) { implRespondToCloseInputFile(fb); }

    void registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper) {
      implRegisterThinnedAssociations(registry, helper);
    }

    void reset() {
      cached_exception_ = std::exception_ptr();
      state_ = Ready;
      waitingTasks_.reset();
      workStarted_ = false;
      numberOfPathsLeftToRun_ = numberOfPathsOn_;
    }

    void postDoEvent(EventPrincipal const&);

    ModuleDescription const& description() const { return *(moduleCallingContext_.moduleDescription()); }
    ModuleDescription const* descPtr() const { return moduleCallingContext_.moduleDescription(); }
    ///The signals are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void setActivityRegistry(std::shared_ptr<ActivityRegistry> areg);

    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);

    //Used to make EDGetToken work
    virtual void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&) = 0;
    virtual void updateLookup(eventsetup::ESRecordsToProxyIndices const&) = 0;
    virtual void resolvePutIndicies(
        BranchType iBranchType,
        std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
            iIndicies) = 0;

    virtual void modulesWhoseProductsAreConsumed(
        std::vector<ModuleDescription const*>& modules,
        ProductRegistry const& preg,
        std::map<std::string, ModuleDescription const*> const& labelsToDesc) const = 0;

    virtual void convertCurrentProcessAlias(std::string const& processName) = 0;

    virtual std::vector<ConsumesInfo> consumesInfo() const = 0;

    virtual Types moduleType() const = 0;

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
    virtual std::string workerType() const = 0;
    virtual bool implDo(EventPrincipal const&, EventSetupImpl const& c, ModuleCallingContext const* mcc) = 0;

    virtual void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual bool implNeedToRunSelection() const = 0;

    virtual void implDoAcquire(EventPrincipal const&,
                               EventSetupImpl const& c,
                               ModuleCallingContext const* mcc,
                               WaitingTaskWithArenaHolder& holder) = 0;

    virtual bool implDoPrePrefetchSelection(StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id,
                                   RunPrincipal const& rp,
                                   EventSetupImpl const& c,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id,
                                 RunPrincipal const& rp,
                                 EventSetupImpl const& c,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(LuminosityBlockPrincipal const& lbp,
                             EventSetupImpl const& c,
                             ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id,
                                   LuminosityBlockPrincipal const& lbp,
                                   EventSetupImpl const& c,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id,
                                 LuminosityBlockPrincipal const& lbp,
                                 EventSetupImpl const& c,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(LuminosityBlockPrincipal const& lbp,
                           EventSetupImpl const& c,
                           ModuleCallingContext const* mcc) = 0;
    virtual void implBeginJob() = 0;
    virtual void implEndJob() = 0;
    virtual void implBeginStream(StreamID) = 0;
    virtual void implEndStream(StreamID) = 0;

    void resetModuleDescription(ModuleDescription const*);

    ActivityRegistry* activityRegistry() { return actReg_.get(); }

  private:
    template <typename T>
    bool runModule(typename T::MyPrincipal const&,
                   EventSetupImpl const& c,
                   StreamID stream,
                   ParentContext const& parentContext,
                   typename T::Context const* context);

    virtual void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;

    virtual std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType) const = 0;

    virtual std::vector<ProductResolverIndex> const& itemsShouldPutInEvent() const = 0;

    virtual void preActionBeforeRunEventAsync(WaitingTask* iTask,
                                              ModuleCallingContext const& moduleCallingContext,
                                              Principal const& iPrincipal) const = 0;

    virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;

    virtual void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) = 0;

    virtual TaskQueueAdaptor serializeRunModule() = 0;

    static void exceptionContext(cms::Exception& ex, ModuleCallingContext const* mcc);

    /*This base class is used to hide the differences between the ID used
     for Event, LuminosityBlock and Run. Using the base class allows us
     to only convert the ID to string form if it is actually needed in
     the call to shouldRethrowException.
     */
    class TransitionIDValueBase {
    public:
      virtual std::string value() const = 0;
      virtual ~TransitionIDValueBase() {}
    };

    template <typename T>
    class TransitionIDValue : public TransitionIDValueBase {
    public:
      TransitionIDValue(T const& iP) : p_(iP) {}
      std::string value() const override {
        std::ostringstream iost;
        iost << p_.id();
        return iost.str();
      }

    private:
      T const& p_;
    };

    bool shouldRethrowException(std::exception_ptr iPtr,
                                ParentContext const& parentContext,
                                bool isEvent,
                                TransitionIDValueBase const& iID) const;

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

    void prefetchAsync(WaitingTask*, ServiceToken const&, ParentContext const& parentContext, Principal const&);

    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    virtual bool hasAcquire() const = 0;

    template <typename T>
    std::exception_ptr runModuleAfterAsyncPrefetch(std::exception_ptr const* iEPtr,
                                                   typename T::MyPrincipal const& ep,
                                                   EventSetupImpl const& es,
                                                   StreamID streamID,
                                                   ParentContext const& parentContext,
                                                   typename T::Context const* context);

    void runAcquire(EventPrincipal const& ep,
                    EventSetupImpl const& es,
                    ParentContext const& parentContext,
                    WaitingTaskWithArenaHolder& holder);

    void runAcquireAfterAsyncPrefetch(std::exception_ptr const* iEPtr,
                                      EventPrincipal const& ep,
                                      EventSetupImpl const& es,
                                      ParentContext const& parentContext,
                                      WaitingTaskWithArenaHolder holder);

    std::exception_ptr handleExternalWorkException(std::exception_ptr const* iEPtr, ParentContext const& parentContext);

    template <typename T>
    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(Worker* worker,
                    typename T::MyPrincipal const& ep,
                    EventSetupImpl const& es,
                    ServiceToken const& token,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    typename T::Context const* context)
          : m_worker(worker),
            m_principal(ep),
            m_es(es),
            m_streamID(streamID),
            m_parentContext(parentContext),
            m_context(context),
            m_serviceToken(token) {}

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

      tbb::task* execute() override {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken);

        //incase the emit causes an exception, we need a memory location
        // to hold the exception_ptr
        std::exception_ptr temp_excptr;
        auto excptr = exceptionPtr();
        if (T::isEvent_ && !m_worker->hasAcquire()) {
          // Caught exception is passed to Worker::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskList
          CMS_SA_ALLOW try {
            //pre was called in prefetchAsync
            m_worker->emitPostModuleEventPrefetchingSignal();
          } catch (...) {
            temp_excptr = std::current_exception();
            if (not excptr) {
              excptr = &temp_excptr;
            }
          }
        }

        if (not excptr) {
          if (auto queue = m_worker->serializeRunModule()) {
            auto const& principal = m_principal;
            auto& es = m_es;
            auto f = [worker = m_worker,
                      &principal,
                      &es,
                      streamID = m_streamID,
                      parentContext = m_parentContext,
                      sContext = m_context,
                      serviceToken = m_serviceToken]() {
              //Need to make the services available
              ServiceRegistry::Operate operateRunModule(serviceToken);

              //If needed, we pause the queue in begin transition and resume it
              // at the end transition. This guarantees that the module
              // only processes one transition at a time
              EnableQueueGuard enableQueueGuard{workerhelper::CallImpl<T>::enableGlobalQueue(worker)};
              std::exception_ptr* ptr = nullptr;
              worker->template runModuleAfterAsyncPrefetch<T>(ptr, principal, es, streamID, parentContext, sContext);
            };
            //keep another global transition from running if necessary
            auto gQueue = workerhelper::CallImpl<T>::pauseGlobalQueue(m_worker);
            if (gQueue) {
              gQueue->push([queue, gQueue, f]() mutable {
                gQueue->pause();
                queue.push(std::move(f));
              });
            } else {
              queue.push(std::move(f));
            }
            return nullptr;
          }
        }

        m_worker->runModuleAfterAsyncPrefetch<T>(excptr, m_principal, m_es, m_streamID, m_parentContext, m_context);
        return nullptr;
      }

    private:
      Worker* m_worker;
      typename T::MyPrincipal const& m_principal;
      EventSetupImpl const& m_es;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      typename T::Context const* m_context;
      ServiceToken m_serviceToken;
    };

    // AcquireTask is only used for the Event case, but we define
    // it as a template so all cases will compile.
    // DUMMY exists to work around the C++ Standard prohibition on
    // fully specializing templates nested in other classes.
    template <typename T, typename DUMMY = void>
    class AcquireTask : public WaitingTask {
    public:
      AcquireTask(Worker* worker,
                  typename T::MyPrincipal const& ep,
                  EventSetupImpl const& es,
                  ServiceToken const& token,
                  ParentContext const& parentContext,
                  WaitingTaskWithArenaHolder holder) {}
      tbb::task* execute() override { return nullptr; }
    };

    template <typename DUMMY>
    class AcquireTask<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>, DUMMY> : public WaitingTask {
    public:
      AcquireTask(Worker* worker,
                  EventPrincipal const& ep,
                  EventSetupImpl const& es,
                  ServiceToken const& token,
                  ParentContext const& parentContext,
                  WaitingTaskWithArenaHolder holder)
          : m_worker(worker),
            m_principal(ep),
            m_es(es),
            m_parentContext(parentContext),
            m_holder(std::move(holder)),
            m_serviceToken(token) {}

      tbb::task* execute() override {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken);

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
            excptr = &temp_excptr;
          }
        }

        if (not excptr) {
          if (auto queue = m_worker->serializeRunModule()) {
            auto const& principal = m_principal;
            auto& es = m_es;
            queue.push([worker = m_worker,
                        &principal,
                        &es,
                        parentContext = m_parentContext,
                        serviceToken = m_serviceToken,
                        holder = m_holder]() {
              //Need to make the services available
              ServiceRegistry::Operate operateRunAcquire(serviceToken);

              std::exception_ptr* ptr = nullptr;
              worker->runAcquireAfterAsyncPrefetch(ptr, principal, es, parentContext, holder);
            });
            return nullptr;
          }
        }

        m_worker->runAcquireAfterAsyncPrefetch(excptr, m_principal, m_es, m_parentContext, std::move(m_holder));
        return nullptr;
      }

    private:
      Worker* m_worker;
      EventPrincipal const& m_principal;
      EventSetupImpl const& m_es;
      ParentContext const m_parentContext;
      WaitingTaskWithArenaHolder m_holder;
      ServiceToken m_serviceToken;
    };

    // This class does nothing unless there is an exception originating
    // in an "External Worker". In that case, it handles converting the
    // exception to a CMS exception and adding context to the exception.
    class HandleExternalWorkExceptionTask : public WaitingTask {
    public:
      HandleExternalWorkExceptionTask(Worker* worker, WaitingTask* runModuleTask, ParentContext const& parentContext);

      tbb::task* execute() override;

    private:
      Worker* m_worker;
      WaitingTask* m_runModuleTask;
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
                       EventPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* /* actReg */,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* /* context*/) {
        //Signal sentry is handled by the module
        return iWorker->implDo(ep, es, mcc);
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
                       RunPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(ep, es, mcc);
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
                       RunPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id, ep, es, mcc);
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
                       RunPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(ep, es, mcc);
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
                       RunPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id, ep, es, mcc);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamRuns(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>> {
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Arg;
      static bool call(Worker* iWorker,
                       StreamID,
                       LuminosityBlockPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(ep, es, mcc);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return iWorker->globalLuminosityBlocksQueue(); }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>> {
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker,
                       StreamID id,
                       LuminosityBlockPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id, ep, es, mcc);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };

    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>> {
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Arg;
      static bool call(Worker* iWorker,
                       StreamID,
                       LuminosityBlockPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(ep, es, mcc);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsGlobalLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker* iWorker) { return iWorker->globalLuminosityBlocksQueue(); }
    };
    template <>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>> {
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> Arg;
      static bool call(Worker* iWorker,
                       StreamID id,
                       LuminosityBlockPrincipal const& ep,
                       EventSetupImpl const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id, ep, es, mcc);
      }
      static bool wantsTransition(Worker const* iWorker) { return iWorker->wantsStreamLuminosityBlocks(); }
      static bool needToRunSelection(Worker const* iWorker) { return false; }
      static SerialTaskQueue* pauseGlobalQueue(Worker* iWorker) { return nullptr; }
      static SerialTaskQueue* enableGlobalQueue(Worker*) { return nullptr; }
    };
  }  // namespace workerhelper

  template <typename T>
  void Worker::doWorkAsync(WaitingTask* task,
                           typename T::MyPrincipal const& ep,
                           EventSetupImpl const& es,
                           ServiceToken const& token,
                           StreamID streamID,
                           ParentContext const& parentContext,
                           typename T::Context const* context) {
    if (not workerhelper::CallImpl<T>::wantsTransition(this)) {
      return;
    }

    waitingTasks_.add(task);
    if (T::isEvent_) {
      timesVisited_.fetch_add(1, std::memory_order_relaxed);
    }

    bool expected = false;
    if (workStarted_.compare_exchange_strong(expected, true)) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      //if have TriggerResults based selection we want to reject the event before doing prefetching
      if (workerhelper::CallImpl<T>::needToRunSelection(this)) {
        //We need to run the selection in a different task so that
        // we can prefetch the data needed for the selection
        auto runTask =
            new (tbb::task::allocate_root()) RunModuleTask<T>(this, ep, es, token, streamID, parentContext, context);

        //make sure the task is either run or destroyed
        struct DestroyTask {
          DestroyTask(edm::WaitingTask* iTask) : m_task(iTask) {}

          ~DestroyTask() {
            auto p = m_task.load();
            if (p) {
              tbb::task::destroy(*p);
            }
          }

          edm::WaitingTask* release() {
            auto t = m_task.load();
            m_task.store(nullptr);
            return t;
          }

          std::atomic<edm::WaitingTask*> m_task;
        };

        auto ownRunTask = std::make_shared<DestroyTask>(runTask);
        auto selectionTask =
            make_waiting_task(tbb::task::allocate_root(),
                              [ownRunTask, parentContext, &ep, token, this](std::exception_ptr const*) mutable {
                                ServiceRegistry::Operate guard(token);
                                prefetchAsync(ownRunTask->release(), token, parentContext, ep);
                              });
        prePrefetchSelectionAsync(selectionTask, token, streamID, &ep);
      } else {
        WaitingTask* moduleTask =
            new (tbb::task::allocate_root()) RunModuleTask<T>(this, ep, es, token, streamID, parentContext, context);
        if (T::isEvent_ && hasAcquire()) {
          WaitingTaskWithArenaHolder runTaskHolder(
              new (tbb::task::allocate_root()) HandleExternalWorkExceptionTask(this, moduleTask, parentContext));
          moduleTask = new (tbb::task::allocate_root())
              AcquireTask<T>(this, ep, es, token, parentContext, std::move(runTaskHolder));
        }
        prefetchAsync(moduleTask, token, parentContext, ep);
      }
    }
  }

  template <typename T>
  std::exception_ptr Worker::runModuleAfterAsyncPrefetch(std::exception_ptr const* iEPtr,
                                                         typename T::MyPrincipal const& ep,
                                                         EventSetupImpl const& es,
                                                         StreamID streamID,
                                                         ParentContext const& parentContext,
                                                         typename T::Context const* context) {
    std::exception_ptr exceptionPtr;
    if (iEPtr) {
      assert(*iEPtr);
      TransitionIDValue<typename T::MyPrincipal> idValue(ep);
      if (shouldRethrowException(*iEPtr, parentContext, T::isEvent_, idValue)) {
        exceptionPtr = *iEPtr;
        setException<T::isEvent_>(exceptionPtr);
      } else {
        setPassed<T::isEvent_>();
      }
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    } else {
      // Caught exception is propagated via WaitingTaskList
      CMS_SA_ALLOW try { runModule<T>(ep, es, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  template <typename T>
  void Worker::doWorkNoPrefetchingAsync(WaitingTask* task,
                                        typename T::MyPrincipal const& principal,
                                        EventSetupImpl const& es,
                                        ServiceToken const& serviceToken,
                                        StreamID streamID,
                                        ParentContext const& parentContext,
                                        typename T::Context const* context) {
    if (not workerhelper::CallImpl<T>::wantsTransition(this)) {
      return;
    }
    waitingTasks_.add(task);
    bool expected = false;
    if (workStarted_.compare_exchange_strong(expected, true)) {
      auto toDo = [this, &principal, &es, streamID, parentContext, context, serviceToken]() {
        std::exception_ptr exceptionPtr;
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          //Need to make the services available
          ServiceRegistry::Operate guard(serviceToken);

          this->runModule<T>(principal, es, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };
      if (auto queue = this->serializeRunModule()) {
        queue.push(toDo);
      } else {
        auto taskToDo = make_functor_task(tbb::task::allocate_root(), toDo);
        tbb::task::spawn(*taskToDo);
      }
    }
  }

  template <typename T>
  bool Worker::doWork(typename T::MyPrincipal const& ep,
                      EventSetupImpl const& es,
                      StreamID streamID,
                      ParentContext const& parentContext,
                      typename T::Context const* context) {
    if (T::isEvent_) {
      timesVisited_.fetch_add(1, std::memory_order_relaxed);
    }
    bool rc = false;

    switch (state_) {
      case Ready:
        break;
      case Pass:
        return true;
      case Fail:
        return false;
      case Exception: {
        std::rethrow_exception(cached_exception_);
      }
    }

    bool expected = false;
    if (not workStarted_.compare_exchange_strong(expected, true)) {
      //another thread beat us here
      auto waitTask = edm::make_empty_waiting_task();
      waitTask->increment_ref_count();

      waitingTasks_.add(waitTask.get());

      waitTask->wait_for_all();

      switch (state_) {
        case Ready:
          assert(false);
        case Pass:
          return true;
        case Fail:
          return false;
        case Exception: {
          std::rethrow_exception(cached_exception_);
        }
      }
    }

    //Need the context to be set until after any exception is resolved
    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    auto resetContext = [](ModuleCallingContext* iContext) {
      iContext->setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    };
    std::unique_ptr<ModuleCallingContext, decltype(resetContext)> prefetchSentry(&moduleCallingContext_, resetContext);

    if (T::isEvent_) {
      //if have TriggerResults based selection we want to reject the event before doing prefetching
      if (workerhelper::CallImpl<T>::needToRunSelection(this)) {
        auto waitTask = edm::make_empty_waiting_task();
        waitTask->set_ref_count(2);
        prePrefetchSelectionAsync(waitTask.get(), ServiceRegistry::instance().presentToken(), streamID, &ep);
        waitTask->decrement_ref_count();
        waitTask->wait_for_all();

        if (state() != Ready) {
          //The selection must have rejected this event
          return true;
        }
      }
      auto waitTask = edm::make_empty_waiting_task();
      {
        //Make sure signal is sent once the prefetching is done
        // [the 'pre' signal was sent in prefetchAsync]
        //The purpose of this block is to send the signal after wait_for_all
        auto sentryFunc = [this](void*) { emitPostModuleEventPrefetchingSignal(); };
        std::unique_ptr<ActivityRegistry, decltype(sentryFunc)> signalSentry(actReg_.get(), sentryFunc);

        //set count to 2 since wait_for_all requires value to not go to 0
        waitTask->set_ref_count(2);

        prefetchAsync(waitTask.get(), ServiceRegistry::instance().presentToken(), parentContext, ep);
        waitTask->decrement_ref_count();
        waitTask->wait_for_all();
      }
      if (waitTask->exceptionPtr() != nullptr) {
        TransitionIDValue<typename T::MyPrincipal> idValue(ep);
        if (shouldRethrowException(*waitTask->exceptionPtr(), parentContext, T::isEvent_, idValue)) {
          setException<T::isEvent_>(*waitTask->exceptionPtr());
          waitingTasks_.doneWaiting(cached_exception_);
          std::rethrow_exception(cached_exception_);
        } else {
          setPassed<T::isEvent_>();
          waitingTasks_.doneWaiting(nullptr);
          return true;
        }
      }
    }

    //successful prefetch so no reset necessary
    prefetchSentry.release();
    if (auto queue = serializeRunModule()) {
      auto serviceToken = ServiceRegistry::instance().presentToken();
      queue.pushAndWait([&]() {
        //Need to make the services available
        ServiceRegistry::Operate guard(serviceToken);
        // This try-catch is primarily for paranoia: runModule() deals internally with exceptions, except for those coming from Service signal actions, which are not supposed to throw exceptions
        CMS_SA_ALLOW try { rc = runModule<T>(ep, es, streamID, parentContext, context); } catch (...) {
        }
      });
    } else {
      // This try-catch is primarily for paranoia: runModule() deals internally with exceptions, except for those coming from Service signal actions, which are not supposed to throw exceptions
      CMS_SA_ALLOW try { rc = runModule<T>(ep, es, streamID, parentContext, context); } catch (...) {
      }
    }
    if (state_ == Exception) {
      waitingTasks_.doneWaiting(cached_exception_);
      std::rethrow_exception(cached_exception_);
    }

    waitingTasks_.doneWaiting(nullptr);
    return rc;
  }

  template <typename T>
  bool Worker::runModule(typename T::MyPrincipal const& ep,
                         EventSetupImpl const& es,
                         StreamID streamID,
                         ParentContext const& parentContext,
                         typename T::Context const* context) {
    //unscheduled producers should advance this
    //if (T::isEvent_) {
    //  ++timesVisited_;
    //}
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    if (T::isEvent_) {
      timesRun_.fetch_add(1, std::memory_order_relaxed);
    }

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallImpl<T>::call(this, streamID, ep, es, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed<T::isEvent_>();
        } else {
          setFailed<T::isEvent_>();
        }
      });
    } catch (cms::Exception& ex) {
      exceptionContext(ex, &moduleCallingContext_);
      TransitionIDValue<typename T::MyPrincipal> idValue(ep);
      if (shouldRethrowException(std::current_exception(), parentContext, T::isEvent_, idValue)) {
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
  std::exception_ptr Worker::runModuleDirectly(typename T::MyPrincipal const& ep,
                                               EventSetupImpl const& es,
                                               StreamID streamID,
                                               ParentContext const& parentContext,
                                               typename T::Context const* context) {
    timesVisited_.fetch_add(1, std::memory_order_relaxed);
    std::exception_ptr const* prefetchingException = nullptr;  // null because there was no prefetching to do
    return runModuleAfterAsyncPrefetch<T>(prefetchingException, ep, es, streamID, parentContext, context);
  }
}  // namespace edm
#endif
