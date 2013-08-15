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
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <memory>
#include <sstream>

namespace edm {
  class EventPrincipal;
  class EarlyDeleteHelper;
  class ProductHolderIndexHelper;
  class StreamID;
  class StreamContext;
  class OutputModuleCommunicator;
  class ParentContext;

  namespace workerhelper {
    template< typename O> class CallImpl;
  }
  class Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    enum Types { kAnalyzer, kFilter, kProducer, kOutputModule};

    Worker(ModuleDescription const& iMD, WorkerParams const& iWP);
    virtual ~Worker();

    Worker(Worker const&) = delete; // Disallow copying and moving
    Worker& operator=(Worker const&) = delete; // Disallow copying and moving

    template <typename T>
    bool doWork(typename T::MyPrincipal&, EventSetup const& c,
		CurrentProcessingContext const* cpc,
                CPUTimer *const timer,
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

    void reset() { state_ = Ready; }

    void pathFinished(EventPrincipal&);
    void postDoEvent(EventPrincipal&);

    ///\return nullptr if Worker is not wrapping an OutputModule
    virtual std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator();
    
    ModuleDescription const& description() const {return md_;}
    ModuleDescription const* descPtr() const {return &md_; }
    ///The signals are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void setActivityRegistry(boost::shared_ptr<ActivityRegistry> areg);
    
    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);
    
    //Used to make EDGetToken work
    virtual void updateLookup(BranchType iBranchType,
                      ProductHolderIndexHelper const&) = 0;

    
    virtual Types moduleType() const =0;

    std::pair<double, double> timeCpuReal() const {
      return std::pair<double, double>(stopwatch_->cpuTime(), stopwatch_->realTime());
    }

    void clearCounters() {
      timesRun_ = timesVisited_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    }
    
    void useStopwatch();

    int timesRun() const { return timesRun_; }
    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    State state() const { return state_; }

    int timesPass() const { return timesPassed(); } // for backward compatibility only - to be removed soon

  protected:
    template<typename O> friend class workerhelper::CallImpl;
    virtual std::string workerType() const = 0;
    virtual bool implDo(EventPrincipal&, EventSetup const& c,
			    CurrentProcessingContext const* cpc,
                            ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(RunPrincipal& rp, EventSetup const& c,
			     CurrentProcessingContext const* cpc,
                             ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id, RunPrincipal& rp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id, RunPrincipal& rp, EventSetup const& c,
                                 CurrentProcessingContext const* cpc,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(RunPrincipal& rp, EventSetup const& c,
                           CurrentProcessingContext const* cpc,
                           ModuleCallingContext const* mcc) = 0;
    virtual bool implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                             CurrentProcessingContext const* cpc,
                             ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamBegin(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc,
                                   ModuleCallingContext const* mcc) = 0;
    virtual bool implDoStreamEnd(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                 CurrentProcessingContext const* cpc,
                                 ModuleCallingContext const* mcc) = 0;
    virtual bool implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                           CurrentProcessingContext const* cpc,
                           ModuleCallingContext const* mcc) = 0;
    virtual void implBeginJob() = 0;
    virtual void implEndJob() = 0;
    virtual void implBeginStream(StreamID) = 0;
    virtual void implEndStream(StreamID) = 0;
    
  private:
    virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;

    virtual void implPreForkReleaseResources() = 0;
    virtual void implPostForkReacquireResources(unsigned int iChildIndex,
                                               unsigned int iNumberOfChildren) = 0;
    RunStopwatch::StopwatchPointer stopwatch_;

    int timesRun_;
    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    State state_;

    ModuleDescription md_;
    ModuleCallingContext moduleCallingContext_;

    ExceptionToActionTable const* actions_; // memory assumed to be managed elsewhere
    boost::shared_ptr<cms::Exception> cached_exception_; // if state is 'exception'

    boost::shared_ptr<ActivityRegistry> actReg_;
    
    EarlyDeleteHelper* earlyDeleteHelper_;
  };

  namespace {
    template <typename T>
    class ModuleSignalSentry {
    public:
      ModuleSignalSentry(ActivityRegistry *a,
                         ModuleDescription& md,
                         typename T::Context const* context,
                         ModuleCallingContext* moduleCallingContext,
                         ParentContext const& parentContext) :
        a_(a), md_(&md), context_(context), moduleCallingContext_(moduleCallingContext) {

        moduleCallingContext_->setContext(ModuleCallingContext::State::kRunning, parentContext);
	if(a_) T::preModuleSignal(a_, md_, context, moduleCallingContext_);
      }

      ~ModuleSignalSentry() {
	if(a_) T::postModuleSignal(a_, md_, context_, moduleCallingContext_);
        moduleCallingContext_->setContext(ModuleCallingContext::State::kInvalid, ParentContext());
      }

    private:
      ActivityRegistry* a_;
      ModuleDescription* md_;
      typename T::Context const* context_;
      ModuleCallingContext* moduleCallingContext_;
    };

    template <typename T>
    void exceptionContext(typename T::MyPrincipal const& principal,
                          cms::Exception& ex,
                          CurrentProcessingContext const* cpc) {
      std::ostringstream ost;
      if (T::isEvent_) {
        ost << "Calling event method";
      }
      else if (T::begin_ && T::branchType_ == InRun) {
        ost << "Calling beginRun";
      }
      else if (T::begin_ && T::branchType_ == InLumi) {
        ost << "Calling beginLuminosityBlock";
      }
      else if (!T::begin_ && T::branchType_ == InLumi) {
        ost << "Calling endLuminosityBlock";
      }
      else if (!T::begin_ && T::branchType_ == InRun) {
        ost << "Calling endRun";
      }
      else {
        // It should be impossible to get here ...
        ost << "Calling unknown function";
      }
      if (cpc && cpc->moduleDescription()) {
        ost << " for module " << cpc->moduleDescription()->moduleName() << "/'" << cpc->moduleDescription()->moduleLabel() << "'";
      }
      ex.addContext(ost.str());
      ost.str("");
      ost << "Running path '";
      if (cpc && cpc->pathName()) {
        ost << *cpc->pathName() << "'";
      }
      else {
        ost << "unknown'";
      }
      ex.addContext(ost.str());
      ost.str("");
      ost << "Processing ";
      ost << principal.id();
      ex.addContext(ost.str());
    }
  }

  namespace workerhelper {
    template<>
    class CallImpl<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>> {
    public:
      static bool call(Worker* iWorker, StreamID, EventPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDo(ep,es,cpc, mcc);
      }
    };
    
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>>{
    public:
      static bool call(Worker* iWorker,StreamID, RunPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoBegin(ep,es,cpc, mcc);
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>>{
    public:
      static bool call(Worker* iWorker,StreamID id, RunPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoStreamBegin(id,ep,es,cpc, mcc);
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>>{
    public:
      static bool call(Worker* iWorker,StreamID, RunPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoEnd(ep,es,cpc, mcc);
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>>{
    public:
      static bool call(Worker* iWorker,StreamID id, RunPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoStreamEnd(id,ep,es,cpc, mcc);
      }
    };
    
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>>{
    public:
      static bool call(Worker* iWorker,StreamID, LuminosityBlockPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoBegin(ep,es,cpc, mcc);
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>>{
    public:
      static bool call(Worker* iWorker,StreamID id, LuminosityBlockPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoStreamBegin(id,ep,es,cpc, mcc);
      }
    };
    
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>>{
    public:
      static bool call(Worker* iWorker,StreamID, LuminosityBlockPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoEnd(ep,es,cpc, mcc);
      }
    };
    template<>
    class CallImpl<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>>{
    public:
      static bool call(Worker* iWorker,StreamID id, LuminosityBlockPrincipal& ep, EventSetup const& es,
                       CurrentProcessingContext const* cpc, ModuleCallingContext const* mcc) {
        return iWorker->implDoStreamEnd(id,ep,es,cpc, mcc);
      }
    };
  }
  
  template <typename T>
  bool Worker::doWork(typename T::MyPrincipal& ep, 
                      EventSetup const& es,
                      CurrentProcessingContext const* cpc,
                      CPUTimer* const iTimer,
                      StreamID streamID,
                      ParentContext const& parentContext,
                      typename T::Context const* context) {

    // A RunStopwatch, but only if we are processing an event.
    RunDualStopwatches stopwatch(T::isEvent_ ? stopwatch_ : RunStopwatch::StopwatchPointer(),
                                 iTimer);

    if (T::isEvent_) {
      ++timesVisited_;
    }
    bool rc = false;

    switch(state_) {
      case Ready: break;
      case Pass: return true;
      case Fail: return false;
      case Exception: {
	  cached_exception_->raise();
      }
    }

    if (T::isEvent_) ++timesRun_;

    try {
      try {
        ModuleSignalSentry<T> cpp(actReg_.get(), md_, context, &moduleCallingContext_, parentContext);
        rc = workerhelper::CallImpl<T>::call(this,streamID,ep,es,cpc, &moduleCallingContext_);

        if (rc) {
          state_ = Pass;
          if (T::isEvent_) ++timesPassed_;
        } else {
          state_ = Fail;
          if (T::isEvent_) ++timesFailed_;
        }
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {

      // NOTE: the warning printed as a result of ignoring or failing
      // a module will only be printed during the full true processing
      // pass of this module

      // Get the action corresponding to this exception.  However, if processing
      // something other than an event (e.g. run, lumi) always rethrow.
      exception_actions::ActionCodes action = (T::isEvent_ ? actions_->find(ex.category()) : exception_actions::Rethrow);

      // If we are processing an endpath and the module was scheduled, treat SkipEvent or FailPath
      // as IgnoreCompletely, so any subsequent OutputModules are still run.
      // For unscheduled modules only treat FailPath as IgnoreCompletely but still allow SkipEvent to throw
      if (cpc && cpc->isEndPath()) {
        if ((action == exception_actions::SkipEvent && !cpc->isUnscheduled()) ||
             action == exception_actions::FailPath) action = exception_actions::IgnoreCompletely;
      }
      switch(action) {
        case exception_actions::IgnoreCompletely:
          rc = true;
          ++timesPassed_;
	  state_ = Pass;
          exceptionContext<T>(ep, ex, cpc);
          edm::printCmsExceptionWarning("IgnoreCompletely", ex);
	  break;
        default:
          if (T::isEvent_) ++timesExcept_;
	  state_ = Exception;
          cached_exception_.reset(ex.clone());
	  cached_exception_->raise();
      }
    }
    return rc;
  }
}
#endif
