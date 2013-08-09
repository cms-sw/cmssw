
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include "boost/bind.hpp"

namespace edm {
  Path::Path(int bitpos, std::string const& path_name,
	     WorkersInPath const& workers,
	     TrigResPtr trptr,
	     ActionTable const& actions,
	     boost::shared_ptr<ActivityRegistry> areg,
	     bool isEndPath,
             StreamContext const* streamContext):
    stopwatch_(),
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
    isEndPath_(isEndPath),
    pathContext_(path_name, bitpos, streamContext) {
  }
  
  bool
  Path::handleWorkerFailure(cms::Exception & e,
			    int nwrwue,
                            bool isEvent,
                            bool begin,
                            BranchType branchType,
                            CurrentProcessingContext const& cpc,
                            std::string const& id) {

    exceptionContext(e, isEvent, begin, branchType, cpc, id);

    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    // If not processing an event, always rethrow.
    actions::ActionCodes action = (isEvent ? act_table_->find(e.category()) : actions::Rethrow);
    switch(action) {
      case actions::FailPath: {
	  should_continue = false;
          edm::printCmsExceptionWarning("FailPath", e);
	  break;
      }
      default: {
	  if (isEvent) ++timesExcept_;
	  state_ = hlt::Exception;
	  recordStatus(nwrwue, isEvent);
	  if (action == actions::Rethrow) {
	    std::string pNF = Exception::codeToString(errors::ProductNotFound);
            if (e.category() == pNF) {
	      std::ostringstream ost;
              ost <<  "If you wish to continue processing events after a " << pNF << " exception,\n" <<
	      "add \"SkipEvent = cms.untracked.vstring('ProductNotFound')\" to the \"options\" PSet in the configuration.\n";
              e.addAdditionalInfo(ost.str());
            }
	  }
          throw;
      }
    }

    return should_continue;
  }

  void
  Path::exceptionContext(cms::Exception & ex,
                         bool isEvent,
                         bool begin,
                         BranchType branchType,
                         CurrentProcessingContext const& cpc,
                         std::string const& id) {
    std::ostringstream ost;
    if (isEvent) {
      ost << "Calling event method";
    }
    else if (begin && branchType == InRun) {
      ost << "Calling beginRun";
    }
    else if (begin && branchType == InLumi) {
      ost << "Calling beginLuminosityBlock";
    }
    else if (!begin && branchType == InLumi) {
      ost << "Calling endLuminosityBlock";
    }
    else if (!begin && branchType == InRun) {
      ost << "Calling endRun";
    }
    else {
      // It should be impossible to get here ...
      ost << "Calling unknown function";
    }
    if (cpc.moduleDescription()) {
      ost << " for module " << cpc.moduleDescription()->moduleName() << "/'" << cpc.moduleDescription()->moduleLabel() << "'";
    }
    ex.addContext(ost.str());
    ost.str("");
    ost << "Running path '";
    if (cpc.pathName()) {
      ost << *cpc.pathName() << "'";
    }
    else {
      ost << "unknown'";
    }
    ex.addContext(ost.str());
    ost.str("");
    ost << "Processing ";
    ost << id;
    ex.addContext(ost.str());
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

  void
  Path::useStopwatch() {
    stopwatch_.reset(new RunStopwatch::StopwatchPointer::element_type);
    for(WorkersInPath::iterator it=workers_.begin(), itEnd = workers_.end();
        it != itEnd;
        ++it) {
      it->useStopwatch();
    }
  }

  void 
  Path::setEarlyDeleteHelpers(std::map<const Worker*,EarlyDeleteHelper*> const& iWorkerToDeleter) {
    //we use a temp so we can overset the size but then when moving to earlyDeleteHelpers we only
    // have to use the space necessary
    std::vector<EarlyDeleteHelper*> temp;
    temp.reserve(iWorkerToDeleter.size());
    for(unsigned int index=0; index !=size();++index) {
      auto found = iWorkerToDeleter.find(getWorker(index));
      if(found != iWorkerToDeleter.end()) {
        temp.push_back(found->second);
        found->second->addedToPath();
      }
    }
    std::vector<EarlyDeleteHelper*> tempCorrectSize(temp.begin(),temp.end());
    earlyDeleteHelpers_.swap(tempCorrectSize);
  }

  void
  Path::handleEarlyFinish(EventPrincipal& iEvent) {
    for(auto helper: earlyDeleteHelpers_) {
      helper->pathFinished(iEvent);
    }
  }

}
