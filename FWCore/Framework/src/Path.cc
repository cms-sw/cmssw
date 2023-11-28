
#include "FWCore/Framework/interface/Path.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/EarlyDeleteHelper.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <algorithm>
#include <sstream>

namespace edm {
  Path::Path(int bitpos,
             std::string const& path_name,
             WorkersInPath const& workers,
             TrigResPtr trptr,
             ExceptionToActionTable const& actions,
             std::shared_ptr<ActivityRegistry> areg,
             StreamContext const* streamContext,
             PathContext::PathType pathType)
      : timesRun_(),
        timesPassed_(),
        timesFailed_(),
        timesExcept_(),
        failedModuleIndex_(workers.size()),
        state_(hlt::Ready),
        bitpos_(bitpos),
        trptr_(trptr),
        actReg_(areg),
        act_table_(&actions),
        workers_(workers),
        pathContext_(path_name, streamContext, bitpos, pathType),
        pathStatusInserter_(nullptr),
        pathStatusInserterWorker_(nullptr) {
    for (auto& workerInPath : workers_) {
      workerInPath.setPathContext(&pathContext_);
    }
    modulesToRun_ = workers_.size();
  }

  Path::Path(Path const& r)
      : timesRun_(r.timesRun_),
        timesPassed_(r.timesPassed_),
        timesFailed_(r.timesFailed_),
        timesExcept_(r.timesExcept_),
        failedModuleIndex_(r.failedModuleIndex_),
        state_(r.state_),
        bitpos_(r.bitpos_),
        trptr_(r.trptr_),
        actReg_(r.actReg_),
        act_table_(r.act_table_),
        workers_(r.workers_),
        pathContext_(r.pathContext_),
        pathStatusInserter_(r.pathStatusInserter_),
        pathStatusInserterWorker_(r.pathStatusInserterWorker_) {
    for (auto& workerInPath : workers_) {
      workerInPath.setPathContext(&pathContext_);
    }
    modulesToRun_ = workers_.size();
  }

  void Path::handleWorkerFailure(cms::Exception& e, int nwrwue, ModuleDescription const& desc, std::string const& id) {
    if (e.context().empty()) {
      exceptionContext(e, true /*isEvent*/, true /*begin*/, InEvent /*branchType*/, desc, id, pathContext_);
    }
    // there is no support as of yet for specific paths having
    // different exception behavior

    // If not processing an event, always rethrow.
    exception_actions::ActionCodes action = act_table_->find(e.category());
    switch (action) {
      case exception_actions::TryToContinue: {
        bool expected = false;
        if (printedException_.compare_exchange_strong(expected, true)) {
          std::ostringstream s;
          s << "Path " << name() << " applying TryToContinue on";
          edm::printCmsExceptionWarning(s.str().c_str(), e);
        }
        break;
      }
      default: {
        if (action == exception_actions::Rethrow) {
          std::string pNF = Exception::codeToString(errors::ProductNotFound);
          if (e.category() == pNF) {
            std::ostringstream ost;
            ost << "If you wish to continue processing events after a " << pNF << " exception,\n"
                << "add \"TryToContinue = cms.untracked.vstring('ProductNotFound')\" to the \"options\" PSet in the "
                   "configuration.\n";
            e.addAdditionalInfo(ost.str());
          }
        }
        //throw will copy which will slice the object
        e.raise();
      }
    }
  }

  void Path::exceptionContext(cms::Exception& ex,
                              bool isEvent,
                              bool begin,
                              BranchType branchType,
                              ModuleDescription const& desc,
                              std::string const& id,
                              PathContext const& pathContext) {
    std::ostringstream ost;
    ost << "Running path '" << pathContext.pathName() << "'";
    ex.addContext(ost.str());
    ost.str("");
    ost << "Processing ";
    //For the event case, the Worker has already
    // added the necessary module context to the exception
    if (begin && branchType == InRun) {
      ost << "stream begin Run";
    } else if (begin && branchType == InLumi) {
      ost << "stream begin LuminosityBlock ";
    } else if (!begin && branchType == InLumi) {
      ost << "stream end LuminosityBlock ";
    } else if (!begin && branchType == InRun) {
      ost << "stream end Run ";
    } else if (isEvent) {
      // It should be impossible to get here ...
      ost << "Event ";
    }
    ost << id;
    ex.addContext(ost.str());
  }

  void Path::threadsafe_setFailedModuleInfo(int nwrwue, bool iExcept) {
    bool expected = false;
    while (stateLock_.compare_exchange_strong(expected, true)) {
      expected = false;
    }
    if (iExcept) {
      if (state_ == hlt::Exception) {
        if (nwrwue < failedModuleIndex_) {
          failedModuleIndex_ = nwrwue;
        }
      } else {
        state_ = hlt::Exception;
        failedModuleIndex_ = nwrwue;
      }
    } else {
      if (state_ != hlt::Exception) {
        if (nwrwue < failedModuleIndex_) {
          failedModuleIndex_ = nwrwue;
        }
        state_ = hlt::Fail;
      }
    }

    stateLock_ = false;
  }

  void Path::recordStatus(int nwrwue, hlt::HLTState state) {
    if (trptr_) {
      trptr_->at(bitpos_) = HLTPathStatus(state, nwrwue);
    }
  }

  void Path::updateCounters(hlt::HLTState state) {
    switch (state) {
      case hlt::Pass: {
        ++timesPassed_;
        break;
      }
      case hlt::Fail: {
        ++timesFailed_;
        break;
      }
      case hlt::Exception: {
        ++timesExcept_;
      }
      default:;
    }
  }

  void Path::clearCounters() {
    using std::placeholders::_1;
    timesRun_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    for_all(workers_, std::bind(&WorkerInPath::clearCounters, _1));
  }

  void Path::setEarlyDeleteHelpers(std::map<const Worker*, EarlyDeleteHelper*> const& iWorkerToDeleter) {
    for (unsigned int index = 0; index != size(); ++index) {
      auto found = iWorkerToDeleter.find(getWorker(index));
      if (found != iWorkerToDeleter.end()) {
        found->second->addedToPath();
      }
    }
  }

  void Path::setPathStatusInserter(PathStatusInserter* pathStatusInserter, Worker* pathStatusInserterWorker) {
    pathStatusInserter_ = pathStatusInserter;
    pathStatusInserterWorker_ = pathStatusInserterWorker;
  }

  void Path::processOneOccurrenceAsync(WaitingTaskHolder iTask,
                                       EventTransitionInfo const& iInfo,
                                       ServiceToken const& iToken,
                                       StreamID const& iStreamID,
                                       StreamContext const* iStreamContext) {
    waitingTasks_.reset();
    modulesToRun_ = workers_.size();
    ++timesRun_;
    waitingTasks_.add(iTask);
    printedException_ = false;
    if (actReg_) {
      ServiceRegistry::Operate guard(iToken);
      actReg_->prePathEventSignal_(*iStreamContext, pathContext_);
    }
    //If the Path succeeds, these are the values we have at the end
    state_ = hlt::Pass;
    failedModuleIndex_ = workers_.size() - 1;

    if (workers_.empty()) {
      ServiceRegistry::Operate guard(iToken);
      finished(std::exception_ptr(), iStreamContext, iInfo, iStreamID);
      return;
    }

    runNextWorkerAsync(0, iInfo, iToken, iStreamID, iStreamContext, *iTask.group());
  }

  void Path::workerFinished(std::exception_ptr const* iException,
                            unsigned int iModuleIndex,
                            EventTransitionInfo const& iInfo,
                            ServiceToken const& iToken,
                            StreamID const& iID,
                            StreamContext const* iContext,
                            oneapi::tbb::task_group& iGroup) {
    EventPrincipal const& iEP = iInfo.principal();
    ServiceRegistry::Operate guard(iToken);

    //This call also allows the WorkerInPath to update statistics
    // so should be done even if an exception happened
    auto& worker = workers_[iModuleIndex];
    bool shouldContinue = worker.checkResultsOfRunWorker(true);
    std::exception_ptr finalException;
    if (iException) {
      shouldContinue = false;
      std::unique_ptr<cms::Exception> pEx;
      try {
        std::rethrow_exception(*iException);
      } catch (cms::Exception& oldEx) {
        pEx = std::unique_ptr<cms::Exception>(oldEx.clone());
      } catch (std::exception const& oldEx) {
        pEx = std::make_unique<edm::Exception>(errors::StdException);
      } catch (...) {
        pEx = std::make_unique<edm::Exception>(errors::Unknown);
      }
      // Caught exception is propagated via WaitingTaskList
      CMS_SA_ALLOW try {
        std::ostringstream ost;
        ost << iEP.id();
        ModuleDescription const* desc = worker.getWorker()->description();
        assert(desc != nullptr);
        handleWorkerFailure(*pEx, iModuleIndex, *desc, ost.str());
        //If we didn't rethrow, then we effectively skipped
        worker.skipWorker(iEP);
      } catch (...) {
        finalException = std::current_exception();
        //set the exception early to avoid case where another Path is waiting
        // on a module in this Path and not running the module will lead to a
        // different but related exception in the other Path. We want this
        // Paths exception to be the one that gets reported.
        waitingTasks_.presetTaskAsFailed(finalException);
      }
    }
    auto const nextIndex = iModuleIndex + 1;
    if (shouldContinue and nextIndex < workers_.size()) {
      if (not worker.runConcurrently()) {
        --modulesToRun_;
        runNextWorkerAsync(nextIndex, iInfo, iToken, iID, iContext, iGroup);
        return;
      }
    }

    if (not shouldContinue) {
      threadsafe_setFailedModuleInfo(iModuleIndex, iException != nullptr);
    }
    if (not shouldContinue and not worker.runConcurrently()) {
      //we are leaving the path early
      for (auto it = workers_.begin() + nextIndex, itEnd = workers_.end(); it != itEnd; ++it) {
        --modulesToRun_;
        it->skipWorker(iEP);
      }
    }
    if (--modulesToRun_ == 0) {
      //The path should only be marked as finished once all outstanding modules finish
      finished(finalException, iContext, iInfo, iID);
    }
  }

  void Path::finished(std::exception_ptr iException,
                      StreamContext const* iContext,
                      EventTransitionInfo const& iInfo,
                      StreamID const& streamID) {
    updateCounters(state_);
    auto failedModuleBitPosition = bitPosition(failedModuleIndex_);
    recordStatus(failedModuleBitPosition, state_);
    // Caught exception is propagated via WaitingTaskList
    CMS_SA_ALLOW try {
      HLTPathStatus status(state_, failedModuleBitPosition);

      if (pathStatusInserter_) {  // pathStatusInserter is null for EndPaths
        pathStatusInserter_->setPathStatus(streamID, status);
      }
      std::exception_ptr jException =
          pathStatusInserterWorker_->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
              iInfo, streamID, ParentContext(iContext), iContext);
      if (jException && not iException) {
        iException = jException;
      }
      actReg_->postPathEventSignal_(*iContext, pathContext_, status);
    } catch (...) {
      if (not iException) {
        iException = std::current_exception();
      }
    }
    waitingTasks_.doneWaiting(iException);
  }

  void Path::runNextWorkerAsync(unsigned int iNextModuleIndex,
                                EventTransitionInfo const& iInfo,
                                ServiceToken const& iToken,
                                StreamID const& iID,
                                StreamContext const* iContext,
                                oneapi::tbb::task_group& iGroup) {
    //Figure out which next modules can run concurrently
    const int firstModuleIndex = iNextModuleIndex;
    int lastModuleIndex = firstModuleIndex;
    while (lastModuleIndex + 1 != static_cast<int>(workers_.size()) and workers_[lastModuleIndex].runConcurrently()) {
      ++lastModuleIndex;
    }
    for (; lastModuleIndex >= firstModuleIndex; --lastModuleIndex) {
      ServiceWeakToken weakToken = iToken;
      auto nextTask = make_waiting_task([this, lastModuleIndex, info = iInfo, iID, iContext, weakToken, &iGroup](
                                            std::exception_ptr const* iException) {
        this->workerFinished(iException, lastModuleIndex, info, weakToken.lock(), iID, iContext, iGroup);
      });
      workers_[lastModuleIndex].runWorkerAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
          WaitingTaskHolder(iGroup, nextTask), iInfo, iToken, iID, iContext);
    }
  }

}  // namespace edm
