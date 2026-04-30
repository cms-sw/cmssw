#ifndef FWCore_Framework_SourceCoordinator_h
#define FWCore_Framework_SourceCoordinator_h
// -*- C++ -*-
// Package:     FWCore/Framework
// Class  :     SourceCoordinator
// Description: This class is responsible for coordinating the source and the state machine
// Original Author:  Chris Jones

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/SourceStatus.h"
#include <memory>
#include <mutex>
#include <tuple>

namespace edm {
  class RunProcessingStatus;
  class LuminosityBlockProcessingStatus;
  class HistoryAppender;
  class EventPrincipal;
  class RunPrincipal;
  class LuminosityBlockPrincipal;
  class WaitingTaskHolder;

  class SourceCoordinator {
  public:
    explicit SourceCoordinator(SharedResourcesAcquirer&& sourceResourcesAcquirer,
                               std::shared_ptr<std::recursive_mutex> sourceMutex,
                               ServiceToken& token)
        : serviceToken_(token),
          sourceResourcesAcquirer_(std::move(sourceResourcesAcquirer)),
          sourceMutex_(sourceMutex) {}

    void setSignals(ActivityRegistry& iRegistry) {
      earlyTerminationSignal_ = &iRegistry.preSourceEarlyTerminationSignal_;
      preSourceNextTransitionSignal_ = &iRegistry.preSourceNextTransitionSignal_;
      postSourceNextTransitionSignal_ = &iRegistry.postSourceNextTransitionSignal_;
    }

    void setSource(std::unique_ptr<InputSource> input) { input_ = std::move(input); }
    void releaseSource() { input_ = nullptr; }
    ProductRegistry const& productRegistry() const { return input_->productRegistry(); }
    void beginJob(ProductRegistry const&);
    void endJob();
    std::tuple<std::shared_ptr<edm::FileBlock>, ProductRegistry const*> readFile();
    void closeFile(FileBlock*, bool cleaningUpAfterException);

    bool randomAccess() const;
    ProcessingController::ForwardState forwardState();
    ProcessingController::ReverseState reverseState();
    void skipEvents(int n);
    bool goToEvent(EventID const& transition);

    void rewind();

    void fillProcessBlockHelper();
    bool nextProcessBlock(ProcessBlockPrincipal& processBlockPrincipal);
    void readProcessBlock(ProcessBlockPrincipal& processBlockPrincipal);

    void peekNextTransitionTypeThatIsNotTheSameRunAsync(RunNumber_t run,
                                                        const ProcessHistoryID& reducedProcessHistoryID,
                                                        SourceStatus& lastTransition,
                                                        WaitingTaskHolder nextTask);

    void readNewRunAsync(std::shared_ptr<RunProcessingStatus> iRunStatus,
                         ProcessContext const& processContext,
                         HistoryAppender& historyAppender,
                         SourceStatus& lastTransition,
                         WaitingTaskHolder iHolder);
    void readNewLuminosityBlockAsync(std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus,
                                     ProcessContext const& processContext,
                                     HistoryAppender& historyAppender,
                                     SourceStatus& lastTransition,
                                     WaitingTaskHolder iHolder);
    struct ReadNextEventForStreamResult {
      //If didCallReadEvent and stopLumi are both false, then we are in the middle of a file transition
      bool didCallReadEvent;
      bool stopLumi;
      InputSource::ItemType nextTransitionType;
      bool mustStartNextLumiOrEndRun = false;
    };

    void readNextEventForStreamAsync(EventPrincipal& event,
                                     ProcessContext& processContext,
                                     LuminosityBlockProcessingStatus& iStatus,
                                     RunProcessingStatus& runStatus,
                                     bool earlierTaskFailed,
                                     bool needToStop,
                                     ReadNextEventForStreamResult& oResult,
                                     SourceStatus& oSourceStatus,
                                     WaitingTaskHolder iTask);

    bool mergeRunIfNeeded(RunPrincipal& iRunPrincipal);
    bool mergeLuminosityBlockIfNeeded(LuminosityBlockPrincipal& iLumiPrincipal);

    [[nodiscard]] SourceStatus thread_unsafe_peekNextTransitionType();

  private:
    void readAndMergeRun(RunPrincipal& iRunPrincipal);
    void readRun(RunPrincipal& iRunPrincipal, HistoryAppender& historyAppender);

    void readLuminosityBlock(LuminosityBlockPrincipal& iLumiPrincipal, HistoryAppender& historyAppender);
    void readAndMergeLuminosityBlock(LuminosityBlockPrincipal& iLumiPrincipal);
    ReadNextEventForStreamResult readNextEventForStream(EventPrincipal& event,
                                                        ProcessContext& processContext,
                                                        LuminosityBlockProcessingStatus& iStatus,
                                                        RunProcessingStatus& runStatus);

    void readEvent(EventPrincipal& event,
                   ProcessContext& processContext,
                   LuminosityBlockProcessingStatus& lumiStatus,
                   RunProcessingStatus& runStatus);
    edm::propagate_const<std::unique_ptr<InputSource>> input_;
    ServiceToken& serviceToken_;
    SharedResourcesAcquirer sourceResourcesAcquirer_;
    std::shared_ptr<std::recursive_mutex> sourceMutex_;
    SourceStatus sourceStatus_;

    ActivityRegistry::PreSourceEarlyTermination* earlyTerminationSignal_ = nullptr;
    ActivityRegistry::PreSourceNextTransition* preSourceNextTransitionSignal_ = nullptr;
    ActivityRegistry::PostSourceNextTransition* postSourceNextTransitionSignal_ = nullptr;
  };
}  // namespace edm
#endif
