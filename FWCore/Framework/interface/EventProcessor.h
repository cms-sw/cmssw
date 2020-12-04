#ifndef FWCore_Framework_EventProcessor_h
#define FWCore_Framework_EventProcessor_h

/*----------------------------------------------------------------------

EventProcessor: This defines the 'framework application' object. It is
configured in the user's main() function, and is set running.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"
#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"

#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <exception>
#include <mutex>

namespace edm {

  class ExceptionToActionTable;
  class BranchIDListHelper;
  class MergeableRunProductMetadata;
  class ThinnedAssociationsHelper;
  class EDLooperBase;
  class HistoryAppender;
  class ProcessDesc;
  class SubProcess;
  class WaitingTaskHolder;
  class LuminosityBlockPrincipal;
  class LuminosityBlockProcessingStatus;
  class IOVSyncValue;

  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;
  }  // namespace eventsetup

  class EventProcessor {
  public:
    // Status codes:
    //   0     successful completion
    //   1     exception of unknown type caught
    //   2     everything else
    //   3     signal received
    //   4     input complete
    //   5     call timed out
    //   6     input count complete
    enum StatusCode {
      epSuccess = 0,
      epException = 1,
      epOther = 2,
      epSignal = 3,
      epInputComplete = 4,
      epTimedOut = 5,
      epCountComplete = 6
    };

    // The input 'parameterSet' contains the entire contents of a  configuration file.
    // Also allows the attachement of pre-existing services specified  by 'token', and
    // the specification of services by name only (defaultServices and forcedServices).
    // 'defaultServices' are overridden by 'parameterSet'.
    // 'forcedServices' the 'parameterSet'.
    explicit EventProcessor(std::unique_ptr<ParameterSet> parameterSet,
                            ServiceToken const& token = ServiceToken(),
                            serviceregistry::ServiceLegacy = serviceregistry::kOverlapIsError,
                            std::vector<std::string> const& defaultServices = std::vector<std::string>(),
                            std::vector<std::string> const& forcedServices = std::vector<std::string>());

    // Same as previous constructor, but without a 'token'.  Token will be defaulted.

    EventProcessor(std::unique_ptr<ParameterSet> parameterSet,
                   std::vector<std::string> const& defaultServices,
                   std::vector<std::string> const& forcedServices = std::vector<std::string>());

    EventProcessor(std::shared_ptr<ProcessDesc> processDesc,
                   ServiceToken const& token,
                   serviceregistry::ServiceLegacy legacy);

    ~EventProcessor();

    EventProcessor(EventProcessor const&) = delete;             // Disallow copying and moving
    EventProcessor& operator=(EventProcessor const&) = delete;  // Disallow copying and moving

    void taskCleanup();

    /**This should be called before the first call to 'run'
       If this is not called in time, it will automatically be called
       the first time 'run' is called
       */
    void beginJob();

    /**This should be called before the EventProcessor is destroyed
       throws if any module's endJob throws an exception.
       */
    void endJob();

    // -------------

    // Same as runToCompletion(false) but since it was used extensively
    // outside of the framework (and is simpler) will keep
    StatusCode run();

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this EventProccessor.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!

    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    ProcessConfiguration const& processConfiguration() const { return *processConfiguration_; }

    /// Return the number of events this EventProcessor has tried to process
    /// (inclues both successes and failures, including failures due
    /// to exceptions during processing).
    int totalEvents() const;

    /// Return the number of events processed by this EventProcessor
    /// which have been passed by one or more trigger paths.
    int totalEventsPassed() const;

    /// Return the number of events that have not passed any trigger.
    /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
    int totalEventsFailed() const;

    /// Turn end_paths "off" if "active" is false;
    /// turn end_paths "on" if "active" is true.
    void enableEndPaths(bool active);

    /// Return true if end_paths are active, and false if they are
    /// inactive.
    bool endPathsEnabled() const;

    /// Clears counters used by trigger report.
    void clearCounters();

    // Really should not be public,
    //   but the EventFilter needs it for now.
    ServiceToken getToken();

    //------------------------------------------------------------------
    //
    // Nested classes and structs below.

    // The function "runToCompletion" will run until the job is "complete",
    // which means:
    //       1 - no more input data
    //       2 - input maxEvents parameter limit reached
    //       3 - output maxEvents parameter limit reached
    //       4 - input maxLuminosityBlocks parameter limit reached
    //       5 - looper directs processing to end
    //
    // The return values from the function are as follows:
    //   epSignal - processing terminated early, SIGUSR2 encountered
    //   epCountComplete - "runEventCount" processed the number of events
    //                     requested by the argument
    //   epSuccess - all other cases
    //
    StatusCode runToCompletion();

    // The following functions are used by the code implementing
    // transition handling.

    InputSource::ItemType nextTransitionType();
    InputSource::ItemType lastTransitionType() const {
      if (deferredExceptionPtrIsSet_) {
        return InputSource::IsStop;
      }
      return lastSourceTransition_;
    }
    std::pair<edm::ProcessHistoryID, edm::RunNumber_t> nextRunID();
    edm::LuminosityBlockNumber_t nextLuminosityBlockID();

    void readFile();
    bool fileBlockValid() { return fb_.get() != nullptr; }
    void closeInputFile(bool cleaningUpAfterException);
    void openOutputFiles();
    void closeOutputFiles();

    void respondToOpenInputFile();
    void respondToCloseInputFile();

    void startingNewLoop();
    bool endOfLoop();
    void rewindInput();
    void prepareForNextLoop();
    bool shouldWeCloseOutput() const;

    void doErrorStuff();

    void beginProcessBlock(bool& beginProcessBlockSucceeded);
    void inputProcessBlocks();
    void endProcessBlock(bool cleaningUpAfterException, bool beginProcessBlockSucceeded);

    void beginRun(ProcessHistoryID const& phid,
                  RunNumber_t run,
                  bool& globalBeginSucceeded,
                  bool& eventSetupForInstanceSucceeded);
    void endRun(ProcessHistoryID const& phid, RunNumber_t run, bool globalBeginSucceeded, bool cleaningUpAfterException);
    void endUnfinishedRun(ProcessHistoryID const& phid,
                          RunNumber_t run,
                          bool globalBeginSucceeded,
                          bool cleaningUpAfterException,
                          bool eventSetupForInstanceSucceeded);

    InputSource::ItemType processLumis(std::shared_ptr<void> const& iRunResource);
    void endUnfinishedLumi();

    void beginLumiAsync(edm::IOVSyncValue const& iSyncValue,
                        std::shared_ptr<void> const& iRunResource,
                        edm::WaitingTaskHolder iHolder);
    void continueLumiAsync(edm::WaitingTaskHolder iHolder);

    void handleEndLumiExceptions(std::exception_ptr const* iPtr, WaitingTaskHolder& holder);
    void globalEndLumiAsync(edm::WaitingTaskHolder iTask, std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus);
    void streamEndLumiAsync(edm::WaitingTaskHolder iTask, unsigned int iStreamIndex);
    std::pair<ProcessHistoryID, RunNumber_t> readRun();
    std::pair<ProcessHistoryID, RunNumber_t> readAndMergeRun();
    void readLuminosityBlock(LuminosityBlockProcessingStatus&);
    int readAndMergeLumi(LuminosityBlockProcessingStatus&);
    using ProcessBlockType = PrincipalCache::ProcessBlockType;
    void writeProcessBlockAsync(WaitingTaskHolder, ProcessBlockType);
    void writeRunAsync(WaitingTaskHolder,
                       ProcessHistoryID const& phid,
                       RunNumber_t run,
                       MergeableRunProductMetadata const*);
    void deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run);
    void writeLumiAsync(WaitingTaskHolder, LuminosityBlockPrincipal& lumiPrincipal);
    void deleteLumiFromCache(LuminosityBlockProcessingStatus&);

    bool shouldWeStop() const;

    void setExceptionMessageFiles(std::string& message);
    void setExceptionMessageRuns(std::string& message);
    void setExceptionMessageLumis();

    bool setDeferredException(std::exception_ptr);

  private:
    //------------------------------------------------------------------
    //
    // Now private functions.
    // init() is used by only by constructors
    void init(std::shared_ptr<ProcessDesc>& processDesc, ServiceToken const& token, serviceregistry::ServiceLegacy);

    bool readNextEventForStream(unsigned int iStreamIndex, LuminosityBlockProcessingStatus& iLumiStatus);

    void handleNextEventForStreamAsync(WaitingTaskHolder iTask, unsigned int iStreamIndex);

    //read the next event using Stream iStreamIndex
    void readEvent(unsigned int iStreamIndex);

    //process the already read event using Stream iStreamIndex
    void processEventAsync(WaitingTaskHolder iHolder, unsigned int iStreamIndex);

    void processEventAsyncImpl(WaitingTaskHolder iHolder, unsigned int iStreamIndex);

    //returns true if an asynchronous stop was requested
    bool checkForAsyncStopRequest(StatusCode&);

    void processEventWithLooper(EventPrincipal&, unsigned int iStreamIndex);

    std::shared_ptr<ProductRegistry const> preg() const { return get_underlying_safe(preg_); }
    std::shared_ptr<ProductRegistry>& preg() { return get_underlying_safe(preg_); }
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
      return get_underlying_safe(branchIDListHelper_);
    }
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<ThinnedAssociationsHelper>& thinnedAssociationsHelper() {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<EDLooperBase const> looper() const { return get_underlying_safe(looper_); }
    std::shared_ptr<EDLooperBase>& looper() { return get_underlying_safe(looper_); }

    void warnAboutModulesRequiringLuminosityBLockSynchronization() const;
    //------------------------------------------------------------------
    //
    // Data members below.
    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.

    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    edm::propagate_const<std::shared_ptr<ProductRegistry>> preg_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    ServiceToken serviceToken_;
    edm::propagate_const<std::unique_ptr<InputSource>> input_;
    InputSource::ItemType lastSourceTransition_;
    edm::propagate_const<std::unique_ptr<eventsetup::EventSetupsController>> espController_;
    edm::propagate_const<std::shared_ptr<eventsetup::EventSetupProvider>> esp_;
    edm::SerialTaskQueue queueWhichWaitsForIOVsToFinish_;
    std::unique_ptr<ExceptionToActionTable const> act_table_;
    std::shared_ptr<ProcessConfiguration const> processConfiguration_;
    ProcessContext processContext_;
    PathsAndConsumesOfModules pathsAndConsumesOfModules_;
    MergeableRunProductProcesses mergeableRunProductProcesses_;
    edm::propagate_const<std::unique_ptr<Schedule>> schedule_;
    std::vector<edm::SerialTaskQueue> streamQueues_;
    std::unique_ptr<edm::LimitedTaskQueue> lumiQueue_;
    std::vector<std::shared_ptr<LuminosityBlockProcessingStatus>> streamLumiStatus_;
    std::atomic<unsigned int> streamLumiActive_{0};  //works as guard for streamLumiStatus

    std::vector<SubProcess> subProcesses_;
    edm::propagate_const<std::unique_ptr<HistoryAppender>> historyAppender_;

    edm::propagate_const<std::unique_ptr<FileBlock>> fb_;
    edm::propagate_const<std::shared_ptr<EDLooperBase>> looper_;

    //The atomic protects concurrent access of deferredExceptionPtr_
    std::atomic<bool> deferredExceptionPtrIsSet_;
    std::exception_ptr deferredExceptionPtr_;

    SharedResourcesAcquirer sourceResourcesAcquirer_;
    std::shared_ptr<std::recursive_mutex> sourceMutex_;
    PrincipalCache principalCache_;
    bool beginJobCalled_;
    bool shouldWeStop_;
    bool fileModeNoMerge_;
    std::string exceptionMessageFiles_;
    std::string exceptionMessageRuns_;
    std::atomic<bool> exceptionMessageLumis_;
    bool forceLooperToEnd_;
    bool looperBeginJobRun_;
    bool forceESCacheClearOnNewRun_;

    PreallocationConfiguration preallocations_;

    bool asyncStopRequestedWhileProcessingEvents_;
    StatusCode asyncStopStatusCodeFromProcessingEvents_;
    bool firstEventInBlock_ = true;

    typedef std::set<std::pair<std::string, std::string>> ExcludedData;
    typedef std::map<std::string, ExcludedData> ExcludedDataMap;
    ExcludedDataMap eventSetupDataToExcludeFromPrefetching_;

    bool printDependencies_ = false;
    bool deleteNonConsumedUnscheduledModules_ = true;
  };  // class EventProcessor

  //--------------------------------------------------------------------

  inline EventProcessor::StatusCode EventProcessor::run() { return runToCompletion(); }
}  // namespace edm
#endif
