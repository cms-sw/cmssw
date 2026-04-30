#include "FWCore/Framework/interface/SourceCoordinator.h"

#include "FWCore/Framework/interface/SourceStatus.h"
#include "FWCore/Framework/src/LuminosityBlockProcessingStatus.h"
#include "FWCore/Framework/src/RunProcessingStatus.h"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include <cassert>
#include <iostream>  //CDJ DEBUGGING
namespace {
  struct SourceNextGuard {
    SourceNextGuard(edm::ActivityRegistry::PreSourceNextTransition* iPre,
                    edm::ActivityRegistry::PostSourceNextTransition* iPost)
        : signal_(iPost) {
      if (iPre) {
        iPre->emit();
      }
    }
    ~SourceNextGuard() noexcept(false) {
      if (signal_)
        signal_->emit();
    }
    edm::ActivityRegistry::PostSourceNextTransition* signal_;
  };

  struct TerminationGuard {
    TerminationGuard(edm::ActivityRegistry::PreSourceEarlyTermination* iSignal) : signal_(iSignal) {}
    void completedSuccessfully() { signal_ = nullptr; }
    ~TerminationGuard() noexcept(false) {
      if (signal_)
        signal_->emit(edm::TerminationOrigin::ExceptionFromThisContext);
    }
    edm::ActivityRegistry::PreSourceEarlyTermination* signal_;
  };
}  // namespace

namespace edm {

  void SourceCoordinator::beginJob(ProductRegistry const& iRegistry) { input_->doBeginJob(iRegistry); }
  void SourceCoordinator::endJob() { input_->doEndJob(); }
  std::tuple<std::shared_ptr<edm::FileBlock>, ProductRegistry const*> SourceCoordinator::readFile() {
    assert(sourceStatus_.nextTransitionType().itemType() == InputSource::ItemType::IsFile);
    auto oldCacheID = input_->productRegistry().cacheIdentifier();
    auto fb = input_->readFile();
    sourceStatus_.setNeedToCallNext(true);
    //incase the input's registry changed
    ProductRegistry const* preg = nullptr;
    if (input_->productRegistry().cacheIdentifier() != oldCacheID) {
      preg = &input_->productRegistry();
    }
    return {std::move(fb), preg};
  }
  void SourceCoordinator::closeFile(FileBlock* fileBlock, bool cleaningUpAfterException) {
    TerminationGuard sentry(earlyTerminationSignal_);
    input_->closeFile(fileBlock, cleaningUpAfterException);
    sentry.completedSuccessfully();
  }

  bool SourceCoordinator::randomAccess() const { return input_->randomAccess(); }
  ProcessingController::ForwardState SourceCoordinator::forwardState() { return input_->forwardState(); }
  ProcessingController::ReverseState SourceCoordinator::reverseState() { return input_->reverseState(); }
  void SourceCoordinator::skipEvents(int n) {
    sourceStatus_.setNeedToCallNext(true);
    input_->skipEvents(n);
  }
  bool SourceCoordinator::goToEvent(EventID const& transition) {
    sourceStatus_.setNeedToCallNext(true);
    return input_->goToEvent(transition);
  }

  void SourceCoordinator::rewind() {
    input_->repeat();
    input_->rewind();
    sourceStatus_.setNeedToCallNext(true);
  }

  void SourceCoordinator::fillProcessBlockHelper() { input_->fillProcessBlockHelper(); }
  bool SourceCoordinator::nextProcessBlock(ProcessBlockPrincipal& processBlockPrincipal) {
    return input_->nextProcessBlock(processBlockPrincipal);
  }
  void SourceCoordinator::readProcessBlock(ProcessBlockPrincipal& processBlockPrincipal) {
    TerminationGuard sentry(earlyTerminationSignal_);
    input_->readProcessBlock(processBlockPrincipal);
    sentry.completedSuccessfully();
  }

  namespace {
    bool checkForAsyncStopRequest() {
      bool returnValue = false;

      // Look for a shutdown signal
      if (edm::shutdown_flag.load(std::memory_order_acquire)) {
        returnValue = true;
        edm::LogSystem("ShutdownSignal") << "an external signal was sent to shutdown the job early.";
        edm::Service<edm::JobReport> jr;
        jr->reportShutdownSignal();
      }
      return returnValue;
    }
  }  // namespace

  SourceStatus SourceCoordinator::thread_unsafe_peekNextTransitionType() {
    if (sourceStatus_.needToCallNext()) {
      TerminationGuard sentry(earlyTerminationSignal_);
      {
        SourceNextGuard guard(preSourceNextTransitionSignal_, postSourceNextTransitionSignal_);
        //For now, do nothing with InputSource::IsSynchronize
        InputSource::ItemTypeInfo itemTypeInfo;
        do {
          itemTypeInfo = input_->nextItemType();
        } while (itemTypeInfo == InputSource::ItemType::IsSynchronize);
        sourceStatus_.setNextTransitionType(itemTypeInfo);
        sourceStatus_.setNeedToCallNext(false);
        if (sourceStatus_.nextTransitionType().itemType() == InputSource::ItemType::IsRun) {
          sourceStatus_.setRunAuxiliary(*input_->runAuxiliary());
          sourceStatus_.setReducedProcessHistoryID(input_->reducedProcessHistoryID());
        } else if (sourceStatus_.nextTransitionType().itemType() == InputSource::ItemType::IsLumi) {
          sourceStatus_.setLuminosityBlockAuxiliary(*input_->luminosityBlockAuxiliary());
          sourceStatus_.setReducedProcessHistoryID(input_->reducedProcessHistoryID());
        }
      }
      sentry.completedSuccessfully();

      if (checkForAsyncStopRequest()) {
        earlyTerminationSignal_->emit(TerminationOrigin::ExternalSignal);
        sourceStatus_.setNextTransitionType(InputSource::ItemType::IsStop);
      }
    }
    return sourceStatus_;
  }
  void SourceCoordinator::peekNextTransitionTypeThatIsNotTheSameRunAsync(RunNumber_t run,
                                                                         const ProcessHistoryID& reducedProcessHistoryID,
                                                                         SourceStatus& lastTransition,
                                                                         WaitingTaskHolder nextTask) {
    auto group = nextTask.group();
    sourceResourcesAcquirer_.serialQueueChain().push(
        *group, [this, run, reducedProcessHistoryID, nextHolder = std::move(nextTask), &lastTransition]() mutable {
          CMS_SA_ALLOW try {
            ServiceRegistry::Operate operate(serviceToken_);
            std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));
            lastTransition = thread_unsafe_peekNextTransitionType();
            if (lastTransition.nextTransitionType() == InputSource::ItemType::IsRun &&
                run == lastTransition.runAuxiliary()->run() &&
                reducedProcessHistoryID == lastTransition.reducedProcessHistoryID()) {
              throw Exception(errors::LogicError)
                  << "InputSource claimed previous Run Entry was last to be merged in this file,\n"
                  << "but the next entry has the same run number and reduced ProcessHistoryID.\n"
                  << "This is probably a bug in the InputSource. Please report to the Core group.\n";
            }
          } catch (...) {
            nextHolder.doneWaiting(std::current_exception());
          }
        });
  }

  void SourceCoordinator::readRun(RunPrincipal& iRunPrincipal, HistoryAppender& historyAppender) {
    iRunPrincipal.setAux(*input_->runAuxiliary());
    {
      TerminationGuard sentry(earlyTerminationSignal_);
      input_->readRun(iRunPrincipal, historyAppender);
      sourceStatus_.setNeedToCallNext(true);
      sentry.completedSuccessfully();
    }
    assert(input_->reducedProcessHistoryID() == iRunPrincipal.reducedProcessHistoryID());
  }
  void SourceCoordinator::readAndMergeRun(RunPrincipal& runPrincipal) {
    runPrincipal.mergeAuxiliary(*input_->runAuxiliary());
    {
      TerminationGuard sentry(earlyTerminationSignal_);
      input_->readAndMergeRun(runPrincipal);
      sourceStatus_.setNeedToCallNext(true);
      sentry.completedSuccessfully();
    }
  }

  void SourceCoordinator::readNewRunAsync(std::shared_ptr<RunProcessingStatus> iRunStatus,
                                          ProcessContext const& processContext,
                                          HistoryAppender& historyAppender,
                                          SourceStatus& lastTransition,
                                          WaitingTaskHolder iHolder) {
    sourceResourcesAcquirer_.serialQueueChain().push(
        *iHolder.group(), [this, iRunStatus, &historyAppender, &processContext, &lastTransition, iHolder]() mutable {
          CMS_SA_ALLOW try {
            ServiceRegistry::Operate operate(serviceToken_);

            {
              std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));
              readRun(*iRunStatus->runPrincipal(), historyAppender);
              sourceStatus_.setNeedToCallNext(true);

              RunPrincipal& runPrincipal = *iRunStatus->runPrincipal();
              {
                TerminationGuard sentry(earlyTerminationSignal_);
                input_->doBeginRun(runPrincipal, &processContext);
                sentry.completedSuccessfully();
              }
              // useful for online processing where can be a long break between Run and Lumi transitions being returned by the InputSource
              if (sourceStatus_.nextTransitionType().itemPosition() == InputSource::ItemPosition::LastItemToBeMerged) {
                lastTransition = sourceStatus_;
                return;
              }
            }
            lastTransition = thread_unsafe_peekNextTransitionType();

            while (lastTransition.nextTransitionType() == InputSource::ItemType::IsRun and
                   iRunStatus->runPrincipal()->run() == lastTransition.runAuxiliary()->run() and
                   iRunStatus->runPrincipal()->reducedProcessHistoryID() == lastTransition.reducedProcessHistoryID()) {
              readAndMergeRun(*iRunStatus->runPrincipal());
              sourceStatus_.setNeedToCallNext(true);
              lastTransition = sourceStatus_;
              if (lastTransition.nextTransitionType().itemPosition() == InputSource::ItemPosition::LastItemToBeMerged) {
                return;
              }
              lastTransition = thread_unsafe_peekNextTransitionType();
            }
          } catch (...) {
            iHolder.doneWaiting(std::current_exception());
          }
        });
  }

  void SourceCoordinator::readLuminosityBlock(LuminosityBlockPrincipal& iLumiPrincipal,
                                              HistoryAppender& historyAppender) {
    iLumiPrincipal.setAux(*input_->luminosityBlockAuxiliary());
    {
      TerminationGuard sentry(earlyTerminationSignal_);
      input_->readLuminosityBlock(iLumiPrincipal, historyAppender);
      sourceStatus_.setNeedToCallNext(true);
      sentry.completedSuccessfully();
    }
  }

  void SourceCoordinator::readAndMergeLuminosityBlock(LuminosityBlockPrincipal& lumiPrincipal) {
    assert(lumiPrincipal.aux().sameIdentity(*input_->luminosityBlockAuxiliary()) or
           input_->processHistoryRegistry().reducedProcessHistoryID(lumiPrincipal.aux().processHistoryID()) ==
               input_->processHistoryRegistry().reducedProcessHistoryID(
                   input_->luminosityBlockAuxiliary()->processHistoryID()));
    lumiPrincipal.mergeAuxiliary(*input_->luminosityBlockAuxiliary());
    {
      TerminationGuard sentry(earlyTerminationSignal_);
      input_->readAndMergeLumi(lumiPrincipal);
      sourceStatus_.setNeedToCallNext(true);
      sentry.completedSuccessfully();
    }
  }

  void SourceCoordinator::readNewLuminosityBlockAsync(std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus,
                                                      ProcessContext const& processContext,
                                                      HistoryAppender& historyAppender,
                                                      SourceStatus& lastTransition,
                                                      WaitingTaskHolder iNextTask) {
    sourceResourcesAcquirer_.serialQueueChain().push(
        *iNextTask.group(),
        [this, iLumiStatus, &processContext, &historyAppender, &lastTransition, iNextTask]() mutable {
          CMS_SA_ALLOW try {
            ServiceRegistry::Operate operate(serviceToken_);

            {
              std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));
              assert(iLumiStatus);
              assert(iLumiStatus->lumiPrincipal());
              readLuminosityBlock(*iLumiStatus->lumiPrincipal(), historyAppender);
              {
                TerminationGuard sentry(earlyTerminationSignal_);
                input_->doBeginLumi(*iLumiStatus->lumiPrincipal(), &processContext);
                sentry.completedSuccessfully();
              }
              // useful for online processing where can be a long break between Run and Lumi transitions being returned by the InputSource
              if (sourceStatus_.nextTransitionType().itemPosition() == InputSource::ItemPosition::LastItemToBeMerged) {
                lastTransition = sourceStatus_;
                return;
              }
              lastTransition = thread_unsafe_peekNextTransitionType();
              while (lastTransition.nextTransitionType() == InputSource::ItemType::IsLumi and
                     iLumiStatus->lumiPrincipal()->luminosityBlock() ==
                         lastTransition.lumiAuxiliary()->luminosityBlock()) {
                readAndMergeLuminosityBlock(*iLumiStatus->lumiPrincipal());
                lastTransition = sourceStatus_;
                if (lastTransition.nextTransitionType().itemPosition() ==
                    InputSource::ItemPosition::LastItemToBeMerged) {
                  return;
                }
                lastTransition = thread_unsafe_peekNextTransitionType();
              }
            }
          } catch (...) {
            iNextTask.doneWaiting(std::current_exception());
          }
        });
  }

  bool SourceCoordinator::mergeRunIfNeeded(RunPrincipal& iRunPrincipal) {
    bool returnValue = false;
    auto lastTransition = thread_unsafe_peekNextTransitionType();

    while (lastTransition.nextTransitionType() == InputSource::ItemType::IsRun and
           iRunPrincipal.run() == lastTransition.runAuxiliary()->run() and
           iRunPrincipal.reducedProcessHistoryID() == lastTransition.reducedProcessHistoryID()) {
      returnValue = true;
      readAndMergeRun(iRunPrincipal);
      sourceStatus_.setNeedToCallNext(true);
      lastTransition = sourceStatus_;
      if (lastTransition.nextTransitionType().itemPosition() == InputSource::ItemPosition::LastItemToBeMerged) {
        return true;
      }
      lastTransition = thread_unsafe_peekNextTransitionType();
    }
    return returnValue;
  }
  bool SourceCoordinator::mergeLuminosityBlockIfNeeded(LuminosityBlockPrincipal& iLumiPrincipal) {
    bool returnValue = false;
    auto lastTransition = thread_unsafe_peekNextTransitionType();
    while (lastTransition.nextTransitionType() == InputSource::ItemType::IsLumi and
           iLumiPrincipal.luminosityBlock() == lastTransition.lumiAuxiliary()->luminosityBlock()) {
      returnValue = true;
      readAndMergeLuminosityBlock(iLumiPrincipal);
      if (lastTransition.nextTransitionType().itemPosition() == InputSource::ItemPosition::LastItemToBeMerged) {
        return true;
      }
      lastTransition = thread_unsafe_peekNextTransitionType();
    }
    return returnValue;
  }

  void SourceCoordinator::readEvent(EventPrincipal& event,
                                    ProcessContext& processContext,
                                    LuminosityBlockProcessingStatus& lumiStatus,
                                    RunProcessingStatus& runStatus) {
    StreamContext streamContext(event.streamID(), &processContext);

    TerminationGuard sentry(earlyTerminationSignal_);
    input_->readEvent(event, streamContext);
    sourceStatus_.setNeedToCallNext(true);

    runStatus.updateLastTimestamp(input_->timestamp());
    lumiStatus.updateLastTimestamp(input_->timestamp());
    sentry.completedSuccessfully();
  }

  SourceCoordinator::ReadNextEventForStreamResult SourceCoordinator::readNextEventForStream(
      EventPrincipal& event,
      ProcessContext& processContext,
      LuminosityBlockProcessingStatus& iStatus,
      RunProcessingStatus& runStatus) {
    // This function returns true if it successfully reads an event for the stream and that
    // requires both that an event is next and there are no problems or requests to stop.

    ServiceRegistry::Operate operate(serviceToken_);

    // need to use lock in addition to the serial task queue because
    // of delayed provenance reading and reading data in response to
    // edm::Refs etc
    std::lock_guard<std::recursive_mutex> guard(*(sourceMutex_.get()));

    InputSource::ItemType itemType = thread_unsafe_peekNextTransitionType().nextTransitionType();

    if (InputSource::ItemType::IsEvent != itemType) {
      // IsFile may continue processing the lumi and
      // looper_ can cause the input source to declare a new IsRun which is actually
      // just a continuation of the previous run
      if (InputSource::ItemType::IsStop == itemType or InputSource::ItemType::IsLumi == itemType or
          (InputSource::ItemType::IsRun == itemType and
           (iStatus.lumiPrincipal()->run() != sourceStatus_.runAuxiliary()->run() or
            iStatus.lumiPrincipal()->runPrincipal().reducedProcessHistoryID() !=
                sourceStatus_.reducedProcessHistoryID()))) {
        if (itemType == InputSource::ItemType::IsLumi &&
            iStatus.lumiPrincipal()->luminosityBlock() == sourceStatus_.lumiAuxiliary()->luminosityBlock()) {
          throw Exception(errors::LogicError)
              << "InputSource claimed previous Lumi Entry was last to be merged in this file,\n"
              << "but the next lumi entry has the same lumi number.\n"
              << "This is probably a bug in the InputSource. Please report to the Core group.\n";
        }
        iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
      } else {
        iStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kPauseForFileTransition);
      }
      return {.didCallReadEvent = false,
              .stopLumi =
                  iStatus.eventProcessingState() == LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi,
              .nextTransitionType = sourceStatus_.nextTransitionType(),
              .mustStartNextLumiOrEndRun = false};
    }
    readEvent(event, processContext, iStatus, runStatus);
    return {.didCallReadEvent = true,
            .stopLumi = false,
            .nextTransitionType = sourceStatus_.nextTransitionType(),
            .mustStartNextLumiOrEndRun = false};
  }

  void SourceCoordinator::readNextEventForStreamAsync(EventPrincipal& event,
                                                      ProcessContext& processContext,
                                                      LuminosityBlockProcessingStatus& iLumiStatus,
                                                      RunProcessingStatus& runStatus,
                                                      bool earlierTaskFailed,
                                                      bool needToStop,
                                                      ReadNextEventForStreamResult& oResult,
                                                      SourceStatus& oSourceStatus,
                                                      WaitingTaskHolder iTask) {
    auto group = iTask.group();
    sourceResourcesAcquirer_.serialQueueChain().push(
        *group,
        [this,
         earlierTaskFailed,
         needToStop,
         iTask = std::move(iTask),
         &event,
         &iLumiStatus,
         &runStatus,
         &oResult,
         &processContext,
         &oSourceStatus]() mutable {
          bool aFailureHappened = earlierTaskFailed;
          if (not earlierTaskFailed) {
            // Did another stream already stop or pause this lumi?
            if (iLumiStatus.eventProcessingState() !=
                LuminosityBlockProcessingStatus::EventProcessingState::kProcessing) {
              oResult = {.didCallReadEvent = false,
                         .stopLumi = iLumiStatus.eventProcessingState() ==
                                     LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi,
                         .nextTransitionType = sourceStatus_.nextTransitionType(),
                         .mustStartNextLumiOrEndRun = false};
            } else if (needToStop) {
              //we need to do this change in the source queue to synchronize sourceStatus_
              sourceStatus_.setNextTransitionType(InputSource::ItemType::IsStop);
              sourceStatus_.setNeedToCallNext(false);
              iLumiStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
              oResult = {.didCallReadEvent = false,
                         .stopLumi = true,
                         .nextTransitionType = InputSource::ItemType::IsStop,
                         .mustStartNextLumiOrEndRun = false};
            } else {
              CMS_SA_ALLOW try {
                ServiceRegistry::Operate operate(serviceToken_);
                oResult = readNextEventForStream(event, processContext, iLumiStatus, runStatus);
              } catch (...) {
                aFailureHappened = true;
                WaitingTaskHolder copyHolder(iTask);
                copyHolder.doneWaiting(std::current_exception());
              }
            }
          }
          if (aFailureHappened) {
            // We want all streams to stop or all streams to pause. If we are already in the
            // middle of pausing streams, then finish pausing all of them and the lumi will be
            // ended later. Otherwise, just end it now.
            if (iLumiStatus.eventProcessingState() ==
                LuminosityBlockProcessingStatus::EventProcessingState::kProcessing) {
              iLumiStatus.setEventProcessingState(LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi);
            }
            oResult = {.didCallReadEvent = false,
                       .stopLumi = true,
                       .nextTransitionType = InputSource::ItemType::IsStop,
                       .mustStartNextLumiOrEndRun = false};
          }

          if (not oResult.didCallReadEvent and
              iLumiStatus.eventProcessingState() == LuminosityBlockProcessingStatus::EventProcessingState::kStopLumi and
              iLumiStatus.startNextLumiOrEndRun()) {
            oResult.mustStartNextLumiOrEndRun = true;
          }
          oSourceStatus = sourceStatus_;
        });
  }

}  // namespace edm
