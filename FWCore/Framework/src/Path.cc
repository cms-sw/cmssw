
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
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
             ExceptionToActionTable const& actions,
             std::shared_ptr<ActivityRegistry> areg,
             StreamContext const* streamContext,
             PathContext::PathType pathType) :
    stopwatch_(),
    timesRun_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(hlt::Ready),
    bitpos_(bitpos),
    trptr_(trptr),
    actReg_(areg),
    act_table_(&actions),
    workers_(workers),
    pathContext_(path_name, streamContext, bitpos, pathType) {

    for (auto& workerInPath : workers_) {
      workerInPath.setPathContext(&pathContext_);
    }
  }

  Path::Path(Path const& r) :
    stopwatch_(r.stopwatch_),
    timesRun_(r.timesRun_),
    timesPassed_(r.timesPassed_),
    timesFailed_(r.timesFailed_),
    timesExcept_(r.timesExcept_),
    state_(r.state_),
    bitpos_(r.bitpos_),
    trptr_(r.trptr_),
    actReg_(r.actReg_),
    act_table_(r.act_table_),
    workers_(r.workers_),
    earlyDeleteHelpers_(r.earlyDeleteHelpers_),
    pathContext_(r.pathContext_) {

    for (auto& workerInPath : workers_) {
      workerInPath.setPathContext(&pathContext_);
    }
  }


  bool
  Path::handleWorkerFailure(cms::Exception & e,
			    int nwrwue,
                            bool isEvent,
                            bool begin,
                            BranchType branchType,
                            ModuleDescription const& desc,
                            std::string const& id) {

    exceptionContext(e, isEvent, begin, branchType, desc, id, pathContext_);

    bool should_continue = true;

    // there is no support as of yet for specific paths having
    // different exception behavior
    
    // If not processing an event, always rethrow.
    exception_actions::ActionCodes action = (isEvent ? act_table_->find(e.category()) : exception_actions::Rethrow);
    switch(action) {
      case exception_actions::FailPath: {
	  should_continue = false;
          edm::printCmsExceptionWarning("FailPath", e);
	  break;
      }
      default: {
	  if (isEvent) ++timesExcept_;
	  state_ = hlt::Exception;
	  recordStatus(nwrwue, isEvent);
	  if (action == exception_actions::Rethrow) {
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
                         ModuleDescription const& desc,
                         std::string const& id,
                         PathContext const& pathContext) {
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
    ost << " for module " << desc.moduleName() << "/'" << desc.moduleLabel() << "'";
    ex.addContext(ost.str());
    ost.str("");
    ost << "Running path '" << pathContext.pathName() << "'";
    ex.addContext(ost.str());
    ost.str("");
    ost << "Processing ";
    ost << id;
    ex.addContext(ost.str());
  }

  void
  Path::recordStatus(int nwrwue, bool isEvent) {
    if(isEvent && trptr_) {
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
