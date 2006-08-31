
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
	     ParameterSet const& proc_pset,
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
  
  void Path::runOneEvent(EventPrincipal& ep, EventSetup const& es)
  {
    RunStopwatch stopwatch(stopwatch_);
    ++timesRun_;
    state_ = edm::hlt::Ready;

    // nwrue =  numWorkersRunWithoutUnhandledException
    int nwrwue = -1;
    bool should_continue = true;
    CurrentProcessingContext cpc(&name_, bitPosition());

    Workers::size_type idx = 0;
    // It seems likely that 'nwrwue' and 'idx' can never differ ---
    // if so, we should remove one of them!.
    for ( Workers::iterator i = workers_.begin(), end = workers_.end();
          i != end && should_continue;
          ++i, ++idx ) {
      ++nwrwue;
      assert ( static_cast<int>(idx) == nwrwue );
      try {
        cpc.activate(idx, i->getWorker()->descPtr());
        should_continue = i->runWorker(ep, es, &cpc);
      }
      catch(cms::Exception& e) {
        // handleWorkerFailure may throw a new exception.
        should_continue = handleWorkerFailure(e, nwrwue);
      }
      catch(...) {
        recordUnknownException(nwrwue);
        throw;
      }
    }
    updateCounters(should_continue);
    recordStatus(nwrwue);
  }

  bool
  Path::handleWorkerFailure(cms::Exception const& e,
			    int nwrwue)
  {
    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    actions::ActionCodes code = act_table_->find(e.rootCause());

    switch(code)
      {
      case actions::IgnoreCompletely:
	{
	  LogWarning(e.category())
	    << "Ignoring Exception in path " << name_
	    << ", message:\n"  << e.what() << "\n";
	  break;
	}
      case actions::FailPath:
	{
	  should_continue = false;
	  LogWarning(e.category())
	    << "Failing path " << name_
	    << ", due to exception, message:\n"
	    << e.what() << "\n";
	  break;
	}
      default:
	{
	  ++timesExcept_;
	  state_ = edm::hlt::Exception;
	  recordStatus(nwrwue);
	  throw edm::Exception(errors::ScheduleExecutionFailure,
			       "ProcessingStopped", e)
	    << "Exception going through path " << name_ << "\n";
	}
      }

    return should_continue;
  }

  void
  Path::recordUnknownException(int nwrwue)
  {
    LogError("PassingThrough")
      << "Exception passing through path " << name_ << "\n";
    ++timesExcept_;
    state_ = edm::hlt::Exception;
    recordStatus(nwrwue);
  }

  void
  Path::recordStatus(int nwrwue)
  {
    (*trptr_)[bitpos_]=HLTPathStatus(state_, nwrwue);    
  }

  void
  Path::updateCounters(bool success)
  {
    if (success)
      {
	++timesPassed_;
	state_ = edm::hlt::Pass;
      }
    else
      {
	++timesFailed_;
	state_ = edm::hlt::Fail;
      }
  }

}
