
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

using namespace std;

namespace edm
{
  Path::Path(int bitpos, const std::string& path_name,
	     const Workers& workers,
	     TrigResPtr trptr,
	     ParameterSet const&,
	     ActionTable& actions,
	     ActivityRegistryPtr areg):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesRun_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(edm::hlt::Ready),
    bitpos_(bitpos),
    name_(path_name),
    trptr_(trptr),
    act_reg_(areg),
    act_table_(&actions),
    workers_(workers)
  {
  }
  
  namespace {
    struct PathSignalSentry {
      PathSignalSentry(const std::string& name,
                         const int& nwrwue,
                         const edm::hlt::HLTState& state,
                         edm::Path::ActivityRegistryPtr areg ):
      name_(name),
      nwrwue_(nwrwue),
      state_(state),
      areg_(areg) {
        areg_->preProcessPathSignal_(name_);
      }
      ~PathSignalSentry() {
        HLTPathStatus status(state_, nwrwue_);
        areg_->postProcessPathSignal_(name_, status);
      }
      
      const std::string& name_;
      const int& nwrwue_;
      const edm::hlt::HLTState& state_;
      edm::Path::ActivityRegistryPtr areg_;
    };
  }
  void Path::runOneEvent(EventPrincipal& ep,
	     EventSetup const& es,
	     BranchActionType const& bat)
  {
    bool const isEvent = (bat == BranchActionEvent);

    //Create the PathSignalSentry before the RunStopwatch so that
    // we only record the time spent in the path not from the signal
    int nwrwue = -1;
    std::auto_ptr<PathSignalSentry> signaler(isEvent? new PathSignalSentry(name_,
                                                                           nwrwue,
                                                                           state_,
                                                                           act_reg_) : 0 );
                                                                           
    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(isEvent ? new RunStopwatch(stopwatch_) : 0);

    if (isEvent) {
      ++timesRun_;
    }
    state_ = edm::hlt::Ready;

    // nwrue =  numWorkersRunWithoutUnhandledException
    bool should_continue = true;
    CurrentProcessingContext cpc(&name_, bitPosition());

    Workers::size_type idx = 0;
    // It seems likely that 'nwrwue' and 'idx' can never differ ---
    // if so, we should remove one of them!.
    for ( Workers::iterator i = workers_.begin(), end = workers_.end();
          i != end && should_continue;
          ++i, ++idx ) {
      ++nwrwue;
      assert (static_cast<int>(idx) == nwrwue);
      try {
        cpc.activate(idx, i->getWorker()->descPtr());
        should_continue = i->runWorker(ep, es, bat, &cpc);
      }
      catch(cms::Exception& e) {
        // handleWorkerFailure may throw a new exception.
        should_continue = handleWorkerFailure(e, nwrwue, isEvent);
      }
      catch(...) {
        recordUnknownException(nwrwue, isEvent);
        throw;
      }
    }
    updateCounters(should_continue, isEvent);
    recordStatus(nwrwue, isEvent);
  }

  bool
  Path::handleWorkerFailure(cms::Exception const& e,
			    int nwrwue, bool isEvent)
  {
    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    actions::ActionCodes code = act_table_->find(e.rootCause());

    switch(code) {
      case actions::IgnoreCompletely: {
	  LogWarning(e.category())
	    << "Ignoring Exception in path " << name_
	    << ", message:\n"  << e.what() << "\n";
	  break;
      }
      case actions::FailPath: {
	  should_continue = false;
	  LogWarning(e.category())
	    << "Failing path " << name_
	    << ", due to exception, message:\n"
	    << e.what() << "\n";
	  break;
      }
      default: {
	  if (isEvent) ++timesExcept_;
	  state_ = edm::hlt::Exception;
	  recordStatus(nwrwue, isEvent);
	  throw edm::Exception(errors::ScheduleExecutionFailure,
			       "ProcessingStopped", e)
	    << "Exception going through path " << name_ << "\n";
      }
    }

    return should_continue;
  }

  void
  Path::recordUnknownException(int nwrwue, bool isEvent)
  {
    LogError("PassingThrough")
      << "Exception passing through path " << name_ << "\n";
    if (isEvent) ++timesExcept_;
    state_ = edm::hlt::Exception;
    recordStatus(nwrwue, isEvent);
  }

  void
  Path::recordStatus(int nwrwue, bool isEvent)
  {
    if(isEvent) {
      (*trptr_)[bitpos_]=HLTPathStatus(state_, nwrwue);    
    }
  }

  void
  Path::updateCounters(bool success, bool isEvent)
  {
    if (success)
      {
	if (isEvent) ++timesPassed_;
	state_ = edm::hlt::Pass;
      }
    else
      {
	if(isEvent) ++timesFailed_;
	state_ = edm::hlt::Fail;
      }
  }

}
