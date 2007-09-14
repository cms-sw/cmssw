
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  Path::Path(int bitpos, std::string const& path_name,
	     WorkersInPath const& workers,
	     TrigResPtr trptr,
	     ParameterSet const&,
	     ActionTable& actions,
	     ActivityRegistryPtr areg,
	     bool isEndPath):
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
    workers_(workers),
    isEndPath_(isEndPath)
  {
  }
  
  bool
  Path::handleWorkerFailure(cms::Exception const& e,
			    int nwrwue, bool isEvent) {
    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    actions::ActionCodes code = act_table_->find(e.rootCause());
    assert (code != actions::FailModule);
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
	  throw e << "Exception going through path " << name_ << "\n";
      }
    }

    return should_continue;
  }

  void
  Path::recordUnknownException(int nwrwue, bool isEvent) {
    LogError("PassingThrough")
      << "Exception passing through path " << name_ << "\n";
    if (isEvent) ++timesExcept_;
    state_ = edm::hlt::Exception;
    recordStatus(nwrwue, isEvent);
  }

  void
  Path::recordStatus(int nwrwue, bool isEvent) {
    if(isEvent) {
      (*trptr_)[bitpos_]=HLTPathStatus(state_, nwrwue);    
    }
  }

  void
  Path::updateCounters(bool success, bool isEvent) {
    if (success) {
      if (isEvent) ++timesPassed_;
      state_ = edm::hlt::Pass;
    } else {
      if(isEvent) ++timesFailed_;
      state_ = edm::hlt::Fail;
    }
  }

}
