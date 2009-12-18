
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include "boost/bind.hpp"

namespace edm {
  Path::Path(int bitpos, std::string const& path_name,
	     WorkersInPath const& workers,
	     TrigResPtr trptr,
	     ActionTable& actions,
	     boost::shared_ptr<ActivityRegistry> areg,
	     bool isEndPath):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesRun_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(hlt::Ready),
    bitpos_(bitpos),
    name_(path_name),
    trptr_(trptr),
    actReg_(areg),
    act_table_(&actions),
    workers_(workers),
    isEndPath_(isEndPath) {
  }
  
  bool
  Path::handleWorkerFailure(cms::Exception const& e,
			    int nwrwue, bool isEvent) {
    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    // If not processing an event, always rethrow.
    actions::ActionCodes action = (isEvent ? act_table_->find(e.rootCause()) : actions::Rethrow);
    assert (action != actions::FailModule);
    switch(action) {
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
	  state_ = hlt::Exception;
	  recordStatus(nwrwue, isEvent);
	  if (action == actions::Rethrow) {
	    std::string pNF = Exception::codeToString(errors::ProductNotFound);
            if (e.category() == pNF) {
              e << "If you wish to continue processing events after a " << pNF << " exception,\n" <<
	      "add \"SkipEvent = cms.untracked.vstring('ProductNotFound')\" to the \"options\" PSet in the configuration.\n";
            }
	  }
          throw Exception(errors::ScheduleExecutionFailure,
              "ProcessingStopped", e)
              << "Exception going through path " << name_ << "\n";
      }
    }

    return should_continue;
  }

  void
  Path::recordUnknownException(int nwrwue, bool isEvent) {
    LogError("PassingThrough")
      << "Exception passing through path " << name_ << "\n";
    if (isEvent) ++timesExcept_;
    state_ = hlt::Exception;
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
      state_ = hlt::Pass;
    } else {
      if(isEvent) ++timesFailed_;
      state_ = hlt::Fail;
    }
  }

  void
  Path::clearCounters() {
    timesRun_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    for_all(workers_, boost::bind(&WorkerInPath::clearCounters, _1));
  }


}
