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
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

namespace edm {
  class EventPrincipal;
  class EarlyDeleteHelper;
  class ProductResolverIndexHelper;
  class ProductResolverIndexAndSkipBit;
  class StreamID;
  class StreamContext;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  namespace workerhelper {
    template< typename O> class CallImpl;
  }

  class Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    enum Types { kAnalyzer, kFilter, kProducer, kOutputModule};

    Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions);
    virtual ~Worker();

    Worker(Worker const&) = delete; // Disallow copying and moving
    Worker& operator=(Worker const&) = delete; // Disallow copying and moving

    template <typename T>
    bool doWork(typename T::MyPrincipal const&, EventSetup const& c,
                StreamID stream,
                ParentContext const& parentContext,
                typename T::Context const* context);
    template <typename T>
    void doWorkAsync(WaitingTask* task,
                     typename T::MyPrincipal const&, EventSetup const& c,
                     StreamID stream,
                     ParentContext const& parentContext,
                     typename T::Context const* context);

    void beginJob() ;
    void endJob();
    void beginStream(StreamID id, StreamContext& streamContext);
    void endStream(StreamID id, StreamContext& streamContext);
    void respondToOpenInputFile(FileBlock const& fb) {implRespondToOpenInputFile(fb);}
    void respondToCloseInputFile(FileBlock const& fb) {implRespondToCloseInputFile(fb);}

    void preForkReleaseResources() {implPreForkReleaseResources();}
    void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {implPostForkReacquireResources(iChildIndex, iNumberOfChildren);}
    void registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper) { implRegisterThinnedAssociations(registry, helper); }

    void reset() {
      cached_exception_ = std::exception_ptr();
      state_ = Ready;
      waitingTasks_.reset();
      workStarted_ = false;
    }

    void pathFinished(EventPrincipal const&);
    void postDoEvent(EventPrincipal const&);

    ModuleDescription const& description() const {return *(moduleCallingContext_.moduleDescription());}
    ModuleDescription const* descPtr() const {return moduleCallingContext_.moduleDescription(); }
    ///The signals are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void setActivityRegistry(std::shared_ptr<ActivityRegistry> areg);

    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);

    //Used to make EDGetToken work
    virtual void updateLookup(BranchType iBranchType,
                      ProductResolverIndexHelper const&) = 0;

    virtual void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                                 ProductRegistry const& preg,
                                                 std::map<std::string, ModuleDescription const*> const& labelsToDesc) const = 0;

    virtual std::vector<ConsumesInfo> consumesInfo() const = 0;

    virtual Types moduleType() const =0;

    void clearCounters() {
      timesRun_.store(0,std::memory_order_relaxed);
      timesVisited_.store(0,std::memory_order_relaxed);
      timesPassed_.store(0,std::memory_order_relaxed);
      timesFailed_.store(0,std::memory_order_relaxed);
      timesExcept_.store(0,std::memory_order_relaxed);
    }

    //NOTE: calling state() is done to force synchronization across threads
    int timesRun() const { return timesRun_.load(std::memory_order_relaxed); }
    int timesVisited() const { return timesVisited_.load(std::memory_order_relaxed); }
    int timesPassed() const { return timesPassed_.load(std::memory_order_relaxed); }
    int timesFailed() const { return timesFailed_.load(std::memory_order_relaxed); }
    int timesExcept() const { return timesExcept_.load(std::memory_order_relaxed); }
    State state() const { return state_; }

    int timesPass() const { return timesPassed(); } // for backward compatibility only - to be removed soon

  protected:
    template<typename O> friend class workerhelper::CallImpl;
    virtual std::string workerType() const = 0;
    virtual bool implDo(EventPrincipal const&, EventSetup const& c,
                        ModuleCallingContext const* mcc) = 0;
    virtual bool implDoPrePrefetchSelection(StreamID id,
                                            EventPrincipal const& ep,
                                            ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(RunPrincipal const& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(RunPrincipal const& rp, EventSetup const& c,
                           ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                             ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                           ModuleCallingContext const* mcc) = 0;
    virtual void implBeginJob() = 0;
    virtual void implEndJob() = 0;
    virtual void implBeginStream(StreamID) = 0;
    virtual void implEndStream(StreamID) = 0;

    void resetModuleDescription(ModuleDescription const*);

    ActivityRegistry* activityRegistry() { return actReg_.get(); }

  private:
    
    template <typename T>
    bool runModule(typename T::MyPrincipal const&, EventSetup const& c,
                StreamID stream,
                ParentContext const& parentContext,
                typename T::Context const* context);

    virtual void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;

    virtual std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFromEvent() const = 0;

    virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;

    virtual void implPreForkReleaseResources() = 0;
    virtual void implPostForkReacquireResources(unsigned int iChildIndex,
                                               unsigned int iNumberOfChildren) = 0;
    virtual void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) = 0;
    
    virtual SerialTaskQueueChain* serializeRunModule() = 0;
    
    static void exceptionContext(const std::string& iID,
                          bool iIsEvent,
                          cms::Exception& ex,
                          ModuleCallingContext const* mcc);
    
    /*This base class is used to hide the differences between the ID used
     for Event, LuminosityBlock and Run. Using the base class allows us
     to only convert the ID to string form if it is actually needed in
     the call to shouldRethrowException.
     */
    class TransitionIDValueBase {
    public:
      virtual std::string value() const = 0;
    };
    
    template< typename T>
    class TransitionIDValue : public TransitionIDValueBase {
    public:
      TransitionIDValue(T const& iP): p_(iP) {}
      virtual std::string value() const override {
        std::ostringstream iost;
        iost<<p_.id();
        return iost.str();
      }
      private:
        T const& p_;
        
    };
    
    bool shouldRethrowException(cms::Exception& ex,
                                ParentContext const& parentContext,
                                bool isEvent,
                                TransitionIDValueBase const& iID) const;

    template<bool IS_EVENT>
    bool setPassed() {
      if(IS_EVENT) {
        timesPassed_.fetch_add(1,std::memory_order_relaxed);
      }
      state_ = Pass;
      return true;
    }

    template<bool IS_EVENT>
    bool setFailed() {
      if(IS_EVENT) {
        timesFailed_.fetch_add(1,std::memory_order_relaxed);
      }
      state_ = Fail;
      return false;
    }

    template<bool IS_EVENT>
    std::exception_ptr setException(std::exception_ptr iException) {
      if (IS_EVENT) {
        timesExcept_.fetch_add(1,std::memory_order_relaxed);
      }
      cached_exception_ = iException; // propagate_const<T> has no reset() function
      state_ = Exception;
      return cached_exception_;
    }
        
    void prefetchAsync(WaitingTask*,
                       ParentContext const& parentContext,
                       Principal const& );
        
    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(),moduleCallingContext_);
    }
    
    template<typename T>
    void runModuleAfterAsyncPrefetch(std::exception_ptr const * iEPtr,
                                     typename T::MyPrincipal const& ep,
                                     EventSetup const& es,
                                     StreamID streamID,
                                     ParentContext const& parentContext,
                                     typename T::Context const* context);
        
    template< typename T>
    class RunModuleTask : public WaitingTask {
    public:
      RunModuleTask(Worker* worker,
                    typename T::MyPrincipal const& ep,
                    EventSetup const& es,
                    StreamID streamID,
                    ParentContext const& parentContext,
                    typename T::Context const* context):
      m_worker(worker),
      m_principal(ep),
      m_es(es),
      m_streamID(streamID),
      m_parentContext(parentContext),
      m_context(context),
      m_serviceToken(ServiceRegistry::instance().presentToken()) {}
      
      tbb::task* execute() override {
        //Need to make the services available early so other services can see them
        ServiceRegistry::Operate guard(m_serviceToken);

        //incase the emit causes an exception, we need a memory location
        // to hold the exception_ptr
        std::exception_ptr temp_excptr;
        auto excptr = exceptionPtr();
        try {
          //pre was called in prefetchAsync
          m_worker->emitPostModuleEventPrefetchingSignal();
        }catch(...) {
          temp_excptr = std::current_exception();
          if(not excptr) {
            excptr = &temp_excptr;
          }
        }

        if( not excptr) {
          if(auto queue = m_worker->serializeRunModule()) {
            Worker* worker = m_worker;
            auto const & principal = m_principal;
            auto& es = m_es;
            auto streamID = m_streamID;
            auto parentContext = m_parentContext;
            auto serviceToken = m_serviceToken;
            auto sContext = m_context;
            queue->push( [worker, &principal, &es, streamID,parentContext,sContext, serviceToken]()
            {
              //Need to make the services available
              ServiceRegistry::Operate guard(serviceToken);

              std::exception_ptr* ptr = nullptr;
              worker->runModuleAfterAsyncPrefetch<T>(ptr,
                                                    principal,
                                                    es,
                                                    streamID,
                                                    parentContext,
                                                    sContext);
            });
            return nullptr;
          }
        }

        m_worker->runModuleAfterAsyncPrefetch<T>(excptr,
                                                 m_principal,
                                                 m_es,
                                                 m_streamID,
                                                 m_parentContext,
                                                 m_context);
        return nullptr;
      }
      
    private:
      Worker* m_worker;
      typename T::MyPrincipal const& m_principal;
      EventSetup const& m_es;
      StreamID m_streamID;
      ParentContext const m_parentContext;
      typename T::Context const* m_context;
      ServiceToken m_serviceToken;
    };
    
    std::atomic<int> timesRun_;
    std::atomic<int> timesVisited_;
    std::atomic<int> timesPassed_;
    std::atomic<int> timesFailed_;
    std::atomic<int> timesExcept_;
    std::atomic<State> state_;

    ModuleCallingContext moduleCallingContext_;

    ExceptionToActionTable const* actions_; // memory assumed to be managed elsewhere
    CMS_THREAD_GUARD(state_) std::exception_ptr cached_exception_; // if state is 'exception'

    std::shared_ptr<ActivityRegistry> actReg_; // We do not use propagate_const because the registry itself is mutable.

    edm::propagate_const<EarlyDeleteHelper*> earlyDeleteHelper_;
    
    edm::WaitingTaskList waitingTasks_;
    std::atomic<bool> workStarted_;
  };

  namespace {
    template <typename T>
    class ModuleSignalSentry {
    public:
      ModuleSignalSentry(ActivityRegistry *a,
                         typename T::Context const* context,
                         ModuleCallingContext const* moduleCallingContext) :
        a_(a), context_(context), moduleCallingContext_(moduleCallingContext) {

	if(a_) T::preModuleSignal(a_, context, moduleCallingContext_);
      }

      ~ModuleSignalSentry() {
	if(a_) T::postModuleSignal(a_, context_, moduleCallingContext_);
      }

    private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      typename T::Context const* context_;
      ModuleCallingContext const* moduleCallingContext_;
    };

  }

  namespace workerhelper {
    template<>
    class CallImpl<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>> {
    public:
      typedef OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker, StreamID,
                       EventPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* /* actReg */,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* /* context*/) {
        //Signal sentry is handled by the module
        return iWorker->implDo(ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return iWorker->implDoPrePrefetchSelection(id,ep,mcc);
      }
    };

    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>>{
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Arg;
      static bool call(Worker* iWorker,StreamID,
                       RunPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>>{
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker,StreamID id,
                       RunPrincipal const & ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id,ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>>{
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Arg;
      static bool call(Worker* iWorker,StreamID,
                       RunPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>>{
    public:
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Arg;
      static bool call(Worker* iWorker,StreamID id,
                       RunPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id,ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };

    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>>{
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Arg;
      static bool call(Worker* iWorker,StreamID,
                       LuminosityBlockPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoBegin(ep,es, mcc);
      }

      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>>{
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Arg;
      static bool call(Worker* iWorker,StreamID id,
                       LuminosityBlockPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamBegin(id,ep,es, mcc);
      }

      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
};

    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>>{
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Arg;
      static bool call(Worker* iWorker,StreamID,
                       LuminosityBlockPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoEnd(ep,es, mcc);
      }
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }

    };
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>>{
    public:
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> Arg;
      static bool call(Worker* iWorker,StreamID id,
                       LuminosityBlockPrincipal const& ep, EventSetup const& es,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        return iWorker->implDoStreamEnd(id,ep,es, mcc);
      }
      
      static bool prePrefetchSelection(Worker* iWorker,StreamID id,
                                       typename Arg::MyPrincipal const& ep,
                                       ModuleCallingContext const* mcc) {
        return true;
      }
    };
  }


  template <typename T>
  void Worker::doWorkAsync(WaitingTask* task,
                           typename T::MyPrincipal const& ep,
                           EventSetup const& es,
                           StreamID streamID,
                           ParentContext const& parentContext,
                           typename T::Context const* context) {
    waitingTasks_.add(task);
    if(T::isEvent_) {
      timesVisited_.fetch_add(1,std::memory_order_relaxed);
    }
    bool expected = false;
    if(workStarted_.compare_exchange_strong(expected,true)) {
      auto runTask = new (tbb::task::allocate_root()) RunModuleTask<T>(
        this, ep,es,streamID,parentContext,context);
      prefetchAsync(runTask, parentContext, ep);
    }
  }
     
  template<typename T>
  void Worker::runModuleAfterAsyncPrefetch(std::exception_ptr const* iEPtr,
                                   typename T::MyPrincipal const& ep,
                                   EventSetup const& es,
                                   StreamID streamID,
                                   ParentContext const& parentContext,
                                   typename T::Context const* context) {
    try {
      convertException::wrap([&]() {
        if(iEPtr) {
          assert(*iEPtr);
          moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid,ParentContext(),nullptr);
          std::rethrow_exception(*iEPtr);
        }

        runModule<T>(ep,es,streamID,parentContext,context);
      });
    } catch( cms::Exception& iException) {
      TransitionIDValue<typename T::MyPrincipal> idValue(ep);
      if(shouldRethrowException(iException, parentContext, T::isEvent_, idValue)) {
        assert(not cached_exception_);
        std::ostringstream iost;
        if(iEPtr) {
          iost<<"Prefetching for unscheduled module ";
        } else {
          iost<<"Calling method for unscheduled module ";
        }
        iost<<description().moduleName() << "/'"
        << description().moduleLabel() << "'";
        iException.addContext(iost.str());
        setException<T::isEvent_>(std::current_exception());
        waitingTasks_.doneWaiting(cached_exception_);
      } else {
        setPassed<T::isEvent_>();
      }
    }
    waitingTasks_.doneWaiting(nullptr);
  }

  template <typename T>
  bool Worker::doWork(typename T::MyPrincipal const& ep,
                      EventSetup const& es,
                      StreamID streamID,
                      ParentContext const& parentContext,
                      typename T::Context const* context) {

    if (T::isEvent_) {
      timesVisited_.fetch_add(1,std::memory_order_relaxed);
    }
    bool rc = false;

    switch(state_) {
      case Ready: break;
      case Pass: return true;
      case Fail: return false;
      case Exception: {
        std::rethrow_exception(cached_exception_);
      }
    }
    
    bool expected = false;
    if(not workStarted_.compare_exchange_strong(expected, true) ) {
      //another thread beat us here
      auto waitTask = edm::make_empty_waiting_task();
      waitTask->increment_ref_count();
      
      waitingTasks_.add(waitTask.get());
      
      waitTask->wait_for_all();
      
      switch(state_) {
        case Ready: assert(false);
        case Pass: return true;
        case Fail: return false;
        case Exception: {
          std::rethrow_exception(cached_exception_);
        }
      }
    }

    //Need the context to be set until after any exception is resolved
    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching,parentContext,nullptr);

    auto resetContext = [](ModuleCallingContext* iContext) {iContext->setContext(ModuleCallingContext::State::kInvalid,ParentContext(),nullptr); };
    std::unique_ptr<ModuleCallingContext, decltype(resetContext)> prefetchSentry(&moduleCallingContext_,resetContext);
    
    try {
      convertException::wrap([&]() {

        if (T::isEvent_) {

          //if have TriggerResults based selection we want to reject the event before doing prefetching
          if( not workerhelper::CallImpl<T>::prePrefetchSelection(this,streamID,ep,&moduleCallingContext_) ) {
            timesRun_.fetch_add(1,std::memory_order_relaxed);
            rc = setPassed<T::isEvent_>();
            waitingTasks_.doneWaiting(nullptr);
            return;
          }
          auto waitTask = edm::make_empty_waiting_task();
          {
            //Make sure signal is sent once the prefetching is done
            // [the 'pre' signal was sent in prefetchAsync]
            //The purpose of this block is to send the signal after wait_for_all
            auto sentryFunc = [this](void*) {
              emitPostModuleEventPrefetchingSignal();
            };
            std::unique_ptr<ActivityRegistry, decltype(sentryFunc)> signalSentry(actReg_.get(),sentryFunc);
            
            //set count to 2 since wait_for_all requires value to not go to 0
            waitTask->set_ref_count(2);

            prefetchAsync(waitTask.get(),parentContext, ep);
            waitTask->decrement_ref_count();
            waitTask->wait_for_all();
          }
          if(waitTask->exceptionPtr() != nullptr) {
            std::rethrow_exception(*(waitTask->exceptionPtr()));
          }
        }
        //successful prefetch so no reset necessary
        prefetchSentry.release();
        if(auto queue = serializeRunModule()) {
          auto serviceToken = ServiceRegistry::instance().presentToken();
          queue->pushAndWait([&]() {
            //Need to make the services available
            ServiceRegistry::Operate guard(serviceToken);
            rc = runModule<T>(ep,es,streamID,parentContext,context);
          });
        } else {
          rc = runModule<T>(ep,es,streamID,parentContext,context);
        }
      });
    }
    catch(cms::Exception& ex) {
      TransitionIDValue<typename T::MyPrincipal> idValue(ep);
      if(shouldRethrowException(ex, parentContext, T::isEvent_, idValue)) {
        assert(not cached_exception_);
        setException<T::isEvent_>(std::current_exception());
        waitingTasks_.doneWaiting(cached_exception_);
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed<T::isEvent_>();
      }
    }
    waitingTasks_.doneWaiting(nullptr);
    return rc;
  }
  
  
  template <typename T>
  bool Worker::runModule(typename T::MyPrincipal const& ep,
                      EventSetup const& es,
                      StreamID streamID,
                      ParentContext const& parentContext,
                      typename T::Context const* context) {
    //unscheduled producers should advance this
    //if (T::isEvent_) {
    //  ++timesVisited_;
    //}
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    if (T::isEvent_) {
      timesRun_.fetch_add(1,std::memory_order_relaxed);
    }
    
    bool rc = workerhelper::CallImpl<T>::call(this,streamID,ep,es, actReg_.get(), &moduleCallingContext_, context);
    
    if (rc) {
      setPassed<T::isEvent_>();
    } else {
      setFailed<T::isEvent_>();
    }
    return rc;
  }
}
#endif
