#ifndef FWCore_Framework_Worker_h
#define FWCore_Framework_Worker_h

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.33 2008/01/15 06:52:00 wmtan Exp $

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/BranchActionType.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };

    Worker(ModuleDescription const& iMD, WorkerParams const& iWP);
    virtual ~Worker();

    template <typename T>
    bool doWork(T&, EventSetup const& c,
		BranchActionType const& bat,
		CurrentProcessingContext const* cpc);
    void beginJob(EventSetup const&) ;
    void endJob();
    void respondToOpenInputFile(FileBlock const& fb) {implRespondToOpenInputFile(fb);}
    void respondToCloseInputFile(FileBlock const& fb) {implRespondToCloseInputFile(fb);}
    void respondToOpenOutputFiles(FileBlock const& fb) {implRespondToOpenOutputFiles(fb);}
    void respondToCloseOutputFiles(FileBlock const& fb) {implRespondToCloseOutputFiles(fb);}

    void reset() { state_ = Ready; }
    
    ModuleDescription const& description() const {return md_;}
    ModuleDescription const* descPtr() const {return &md_; }
    ///The signals passed in are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void connect(ActivityRegistry::PreModule&,
                 ActivityRegistry::PostModule&,
                 ActivityRegistry::PreModuleBeginJob&,
                 ActivityRegistry::PostModuleBeginJob&,
                 ActivityRegistry::PreModuleEndJob&,
                 ActivityRegistry::PostModuleEndJob&);

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    void clearCounters() {
      timesRun_ = timesVisited_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    }

    int timesRun() const { return timesRun_; }
    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    State state() const { return state_; }
   
    struct Sigs {
      Sigs();
      ActivityRegistry::PreModule* preModuleSignal;
      ActivityRegistry::PostModule* postModuleSignal;
      ActivityRegistry::PreModuleBeginJob* preModuleBeginJobSignal;
      ActivityRegistry::PostModuleBeginJob* postModuleBeginJobSignal;
      ActivityRegistry::PreModuleEndJob* preModuleEndJobSignal;
      ActivityRegistry::PostModuleEndJob* postModuleEndJobSignal;
    };

    int timesPass() const { return timesPassed(); } // for backward compatibility only - to be removed soon

    class CallPrePost {
    public:
      CallPrePost(EventPrincipal const& , Worker::Sigs& s, ModuleDescription& md) : s_(&s), md_(&md) {
	(*(s_->preModuleSignal))(*md_);
      }
      CallPrePost(LuminosityBlockPrincipal const& , Worker::Sigs&, ModuleDescription& md) : s_(0), md_(&md) {}
      CallPrePost(RunPrincipal const& , Worker::Sigs&, ModuleDescription& md):s_(0),md_(&md) {}
      ~CallPrePost() {
	if (s_ != 0) (*(s_->postModuleSignal))(*md_);
      }
    private:
      Worker::Sigs* s_;
      ModuleDescription* md_;
    };
  protected:
    virtual std::string workerType() const = 0;
    virtual bool implDoWork(EventPrincipal&, EventSetup const& c, 
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc) = 0;
    virtual bool implDoWork(RunPrincipal& rp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc) = 0;
    virtual bool implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc) = 0;
    virtual void implBeginJob(EventSetup const&) = 0;
    virtual void implEndJob() = 0;

  private:
    virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb) = 0;
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb) = 0;

    RunStopwatch::StopwatchPointer stopwatch_;

    int timesRun_;
    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    State state_;

    ModuleDescription md_;
    ActionTable const* actions_; // memory assumed to be managed elsewhere
    boost::shared_ptr<edm::Exception> cached_exception_; // if state is 'exception'

    Sigs sigs_;
  };

  namespace {
    template <typename T>
    cms::Exception& exceptionContext(ModuleDescription const& iMD,
				     T const& ip,
				     cms::Exception& iEx) {
      iEx << iMD.moduleName_ << "/" << iMD.moduleLabel_ 
        << " " << ip.id() << "\n";
      return iEx;
    }
  }

   template <typename T>
   bool Worker::doWork(T& ep, EventSetup const& es,
		      BranchActionType const& bat,
		      CurrentProcessingContext const* cpc) {

    bool const isEvent = (bat == BranchActionEvent);

    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(isEvent ? new RunStopwatch(stopwatch_) : 0);

    if (isEvent) {
      ++timesVisited_;
    }
    bool rc = false;

    switch(state_) {
      case Ready: break;
      case Pass: return true;
      case Fail: return false;
      case Exception: {
	  // rethrow the cached exception again
	  // It seems impossible to
	  // get here a second time until a cms::Exception has been 
	  // thrown prviously.
	  LogWarning("repeat") << "A module has been invoked a second "
			       << "time even though it caught an "
			       << "exception during the previous "
			       << "invocation.\n"
			       << "This may be an indication of a "
			       << "configuration problem.\n";

	  throw *cached_exception_;
      }
    }

    if (isEvent) ++timesRun_;

    try {

	CallPrePost cpp(ep, sigs_, md_);
	rc = implDoWork(ep, es, bat, cpc);

	if (rc) {
	  state_ = Pass;
	  if (isEvent) ++timesPassed_;
	} else {
	  state_ = Fail;
	  if (isEvent) ++timesFailed_;
	}
    }

    catch(cms::Exception& e) {
      
	// NOTE: the warning printed as a result of ignoring or failing
	// a module will only be printed during the full true processing
	// pass of this module

	// Get the action corresponding to this exception.  However, if processing
	// something other than an event (e.g. run, lumi) always rethrow.
	actions::ActionCodes action = (isEvent ? actions_->find(e.rootCause()) : actions::Rethrow);

	// If we are processing an endpath, treat SkipEvent or FailPath
	// as FailModule, so any subsequent OutputModules are still run.
	if (cpc && cpc->isEndPath()) {
	  if (action == actions::SkipEvent || action == actions::FailPath) action = actions::FailModule;
	}
	switch(action) {
	  case actions::IgnoreCompletely: {
	      rc=true;
	      ++timesPassed_;
	      state_ = Pass;
	      LogWarning("IgnoreCompletely")
		<< "Module ignored an exception\n"
                <<e.what()<<"\n";
	      break;
	  }

	  case actions::FailModule: {
	      rc=true;
	      LogWarning("FailModule")
                << "Module failed due to an exception\n"
                << e.what() << "\n";
	      ++timesFailed_;
	      state_ = Fail;
	      break;
	  }
	    
	  default: {

	      // we should not need to include the event/run/module names
	      // the exception because the error logger will pick this
	      // up automatically.  I'm leaving it in until this is 
	      // verified

	      // here we simply add a small amount of data to the
	      // exception to add some context, we could have rethrown
	      // it as something else and embedded with this exception
	      // as an argument to the constructor.

	      if (isEvent) ++timesExcept_;
	      state_ = Exception;
	      e << "cms::Exception going through module ";
              exceptionContext(md_, ep, e);
	      edm::Exception *edmEx = dynamic_cast<edm::Exception *>(&e);
	      if (edmEx) {
	        cached_exception_.reset(new edm::Exception(*edmEx));
	      } else {
	        cached_exception_.reset(new edm::Exception(errors::OtherCMS, std::string(), e));
	      }
	      throw;
	  }
	}
      }
    
    catch(std::bad_alloc& bda) {
	if (isEvent) ++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadAlloc));
	*cached_exception_
	  << "A std::bad_alloc exception occurred during a call to the module ";
	exceptionContext(md_, ep, *cached_exception_)
	  << "The job has probably exhausted the virtual memory available to the process.\n";
	throw *cached_exception_;
    }
    catch(std::exception& e) {
	if (isEvent) ++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::StdException));
	*cached_exception_
	  << "A std::exception occurred during a call to the module ";
        exceptionContext(md_, ep, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n" << e.what();
	throw *cached_exception_;
    }
    catch(std::string& s) {
	if (isEvent) ++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "std::string"));
	*cached_exception_
	  << "A std::string thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, ep, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n string = " << s;
	throw *cached_exception_;
    }
    catch(char const* c) {
	if (isEvent) ++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "const char *"));
	*cached_exception_
	  << "A const char* thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, ep, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n const char* = " << c << "\n";
	throw *cached_exception_;
    }
    catch(...) {
	if (isEvent) ++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::Unknown, "repeated"));
	*cached_exception_
	  << "An unknown occurred during a previous call to the module ";
        exceptionContext(md_, ep, *cached_exception_) << "and cannot be repropagated.\n";
	throw *cached_exception_;
    }

    return rc;
  }

}
#endif
