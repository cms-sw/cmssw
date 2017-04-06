
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

namespace edm {
  Path::Path(int bitpos, std::string const& path_name,
             WorkersInPath const& workers,
             TrigResPtr trptr,
             ExceptionToActionTable const& actions,
             std::shared_ptr<ActivityRegistry> areg,
             StreamContext const* streamContext,
             std::atomic<bool>* stopProcessingEvent,
             PathContext::PathType pathType) :
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
    pathContext_(path_name, streamContext, bitpos, pathType),
    stopProcessingEvent_(stopProcessingEvent){

    for (auto& workerInPath : workers_) {
      workerInPath.setPathContext(&pathContext_);
    }
  }

  Path::Path(Path const& r) :
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
    pathContext_(r.pathContext_),
    stopProcessingEvent_(r.stopProcessingEvent_){

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
      case exception_actions::SkipEvent: {
        //Need the other Paths to stop as soon as possible
        if(stopProcessingEvent_) {
          *stopProcessingEvent_ = true;
        }
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
          throw e;
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
    if (not isEvent) {
      //For the event case, the Worker has already
      // added the necessary module context to the exception
      if (begin && branchType == InRun) {
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
    }
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
    using std::placeholders::_1;
    timesRun_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    for_all(workers_, std::bind(&WorkerInPath::clearCounters, _1));
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
  Path::handleEarlyFinish(EventPrincipal const& iEvent) {
    for(auto helper: earlyDeleteHelpers_) {
      helper->pathFinished(iEvent);
    }
  }

  void
  Path::processOneOccurrenceAsync(WaitingTask* iTask,
                                  EventPrincipal const& iEP,
                                  EventSetup const& iES,
                                  StreamID const& iStreamID,
                                  StreamContext const* iStreamContext) {
    waitingTasks_.reset();
    ++timesRun_;
    waitingTasks_.add(iTask);
    if(actReg_) {
      actReg_->prePathEventSignal_(*iStreamContext, pathContext_);
    }
    state_ = hlt::Ready;
    
    if(workers_.empty()) {
      finished(-1, true, std::exception_ptr(), iStreamContext);
      return;
    }
    
    runNextWorkerAsync(0,iEP,iES,iStreamID, iStreamContext);
  }

  void
  Path::workerFinished(std::exception_ptr const* iException,
                       unsigned int iModuleIndex,
                       EventPrincipal const& iEP, EventSetup const& iES,
                       StreamID const& iID, StreamContext const* iContext) {
    
    //This call also allows the WorkerInPath to update statistics
    // so should be done even if an exception happened
    auto& worker = workers_[iModuleIndex];
    bool shouldContinue = worker.checkResultsOfRunWorker(true);
    std::exception_ptr finalException;
    if(iException) {
      std::unique_ptr<cms::Exception> pEx;
      try {
        std::rethrow_exception(*iException);
      } catch(cms::Exception& oldEx) {
        pEx = std::make_unique<cms::Exception>(oldEx);
      }
      try {
        std::ostringstream ost;
        ost << iEP.id();
        shouldContinue = handleWorkerFailure(*pEx, iModuleIndex, /*isEvent*/ true, /*isBegin*/ true, InEvent,
                                              worker.getWorker()->description(), ost.str());
        //If we didn't rethrow, then we effectively skipped
        worker.skipWorker(iEP);
        finalException = std::exception_ptr();
      } catch(...) {
        shouldContinue = false;
        finalException = std::current_exception();
      }
    }
    if(stopProcessingEvent_ and *stopProcessingEvent_) {
      shouldContinue = false;
    }
    auto const nextIndex = iModuleIndex +1;
    if (shouldContinue and nextIndex < workers_.size()) {
      runNextWorkerAsync(nextIndex, iEP, iES, iID, iContext);
      return;
    }
    
    if (not shouldContinue) {
      //we are leaving the path early
      for(auto it = workers_.begin()+nextIndex, itEnd=workers_.end();
          it != itEnd; ++it) {
        it->skipWorker(iEP);
      }
      handleEarlyFinish(iEP);
    }
    finished(iModuleIndex, shouldContinue, finalException, iContext);
  }
  
  void
  Path::finished(int iModuleIndex, bool iSucceeded, std::exception_ptr iException, StreamContext const* iContext) {
    
    if(not iException) {
      updateCounters(iSucceeded, true);
      recordStatus(iModuleIndex, true);
    }
    try {
      HLTPathStatus status(state_, iModuleIndex);
      actReg_->postPathEventSignal_(*iContext, pathContext_, status);
    } catch(...) {
      if(not iException) {
        iException = std::current_exception();
      }
    }
    waitingTasks_.doneWaiting(iException);
  }
  
  void
  Path::runNextWorkerAsync(unsigned int iNextModuleIndex,
                           EventPrincipal const& iEP, EventSetup const& iES,
                           StreamID const& iID, StreamContext const* iContext) {
    
    //need to make sure Service system is activated on the reading thread
    auto token = ServiceRegistry::instance().presentToken();

    auto nextTask = make_waiting_task( tbb::task::allocate_root(),
                                      [this, iNextModuleIndex, &iEP,&iES, iID, iContext, token](std::exception_ptr const* iException)
    {
      ServiceRegistry::Operate guard(token);
      this->workerFinished(iException, iNextModuleIndex, iEP,iES,iID,iContext);
    });
    
    workers_[iNextModuleIndex].runWorkerAsync<
    OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(nextTask,
                                                                iEP,
                                                                iES,
                                                                iID,
                                                                iContext);
  }

}
