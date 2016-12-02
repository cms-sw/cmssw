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
#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Framework/interface/InputSource.h"
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

#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include "boost/thread/condition.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <exception>
#include <mutex>

namespace statemachine {
  class Machine;
  class Run;
}

namespace edm {

  class ExceptionToActionTable;
  class BranchIDListHelper;
  class ThinnedAssociationsHelper;
  class EDLooperBase;
  class HistoryAppender;
  class ProcessDesc;
  class SubProcess;
  class WaitingTaskHolder;
  class WaitingTask;
  
  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;
  }

  class EventProcessor : public IEventProcessor {
  public:

    // The input string 'config' contains the entire contents of a  configuration file.
    // Also allows the attachement of pre-existing services specified  by 'token', and
    // the specification of services by name only (defaultServices and forcedServices).
    // 'defaultServices' are overridden by 'config'.
    // 'forcedServices' override the 'config'.
    explicit EventProcessor(std::string const& config,
                            ServiceToken const& token = ServiceToken(),
                            serviceregistry::ServiceLegacy = serviceregistry::kOverlapIsError,
                            std::vector<std::string> const& defaultServices = std::vector<std::string>(),
                            std::vector<std::string> const& forcedServices = std::vector<std::string>());

    // Same as previous constructor, but without a 'token'.  Token will be defaulted.

    EventProcessor(std::string const& config,
                   std::vector<std::string> const& defaultServices,
                   std::vector<std::string> const& forcedServices = std::vector<std::string>());

    EventProcessor(std::shared_ptr<ProcessDesc>& processDesc,
                   ServiceToken const& token,
                   serviceregistry::ServiceLegacy legacy);

    /// meant for unit tests
    EventProcessor(std::string const& config, bool isPython);

    ~EventProcessor();

    EventProcessor(EventProcessor const&) = delete; // Disallow copying and moving
    EventProcessor& operator=(EventProcessor const&) = delete; // Disallow copying and moving

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

    std::vector<ModuleDescription const*>
    getAllModuleDescriptions() const;

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

    /// Return the trigger report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerReport(TriggerReport& rep) const;

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
    virtual StatusCode runToCompletion() override;

    // The following functions are used by the code implementing our
    // boost statemachine

    virtual void readFile() override;
    virtual void closeInputFile(bool cleaningUpAfterException) override;
    virtual void openOutputFiles() override;
    virtual void closeOutputFiles() override;

    virtual void respondToOpenInputFile() override;
    virtual void respondToCloseInputFile() override;

    virtual void startingNewLoop() override;
    virtual bool endOfLoop() override;
    virtual void rewindInput() override;
    virtual void prepareForNextLoop() override;
    virtual bool shouldWeCloseOutput() const override;

    virtual void doErrorStuff() override;

    virtual void beginRun(statemachine::Run const& run) override;
    virtual void endRun(statemachine::Run const& run, bool cleaningUpAfterException) override;

    virtual void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException) override;

    virtual statemachine::Run readRun() override;
    virtual statemachine::Run readAndMergeRun() override;
    virtual int readLuminosityBlock() override;
    virtual int readAndMergeLumi() override;
    virtual void writeRun(statemachine::Run const& run) override;
    virtual void deleteRunFromCache(statemachine::Run const& run) override;
    virtual void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;

    virtual void readAndProcessEvent() override;
    virtual bool shouldWeStop() const override;

    virtual void setExceptionMessageFiles(std::string& message) override;
    virtual void setExceptionMessageRuns(std::string& message) override;
    virtual void setExceptionMessageLumis(std::string& message) override;

    virtual bool alreadyHandlingException() const override;

    //returns 'true' if this was a child and we should continue processing
    bool forkProcess(std::string const& jobReportFile);

  private:
    //------------------------------------------------------------------
    //
    // Now private functions.
    // init() is used by only by constructors
    void init(std::shared_ptr<ProcessDesc>& processDesc,
              ServiceToken const& token,
              serviceregistry::ServiceLegacy);

    void terminateMachine(std::unique_ptr<statemachine::Machine>);
    std::unique_ptr<statemachine::Machine> createStateMachine();

    void setupSignal();

    void possiblyContinueAfterForkChildFailure();
    
    bool readNextEventForStream(unsigned int iStreamIndex,
                                     std::atomic<bool>* finishedProcessingEvents);

    void handleNextEventForStreamAsync(WaitingTask* iTask,
                                       unsigned int iStreamIndex,
                                     std::atomic<bool>* finishedProcessingEvents);

    
    //read the next event using Stream iStreamIndex
    void readEvent(unsigned int iStreamIndex);

    //process the already read event using Stream iStreamIndex
    void processEventAsync(WaitingTaskHolder iHolder,
                           unsigned int iStreamIndex);

    //returns true if an asynchronous stop was requested
    bool checkForAsyncStopRequest(StatusCode&);
    
    void processEventWithLooper(EventPrincipal&);

    std::shared_ptr<ProductRegistry const> preg() const {return get_underlying_safe(preg_);}
    std::shared_ptr<ProductRegistry>& preg() {return get_underlying_safe(preg_);}
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {return get_underlying_safe(branchIDListHelper_);}
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() {return get_underlying_safe(branchIDListHelper_);}
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {return get_underlying_safe(thinnedAssociationsHelper_);}
    std::shared_ptr<ThinnedAssociationsHelper>& thinnedAssociationsHelper() {return get_underlying_safe(thinnedAssociationsHelper_);}
    std::shared_ptr<EDLooperBase const> looper() const {return get_underlying_safe(looper_);}
    std::shared_ptr<EDLooperBase>& looper() {return get_underlying_safe(looper_);}
    //------------------------------------------------------------------
    //
    // Data members below.
    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.

    std::shared_ptr<ActivityRegistry> actReg_; // We do not use propagate_const because the registry itself is mutable.
    edm::propagate_const<std::shared_ptr<ProductRegistry>> preg_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    ServiceToken                                  serviceToken_;
    edm::propagate_const<std::unique_ptr<InputSource>> input_;
    edm::propagate_const<std::unique_ptr<eventsetup::EventSetupsController>> espController_;
    edm::propagate_const<std::shared_ptr<eventsetup::EventSetupProvider>> esp_;
    std::unique_ptr<ExceptionToActionTable const>          act_table_;
    std::shared_ptr<ProcessConfiguration const>       processConfiguration_;
    ProcessContext                                processContext_;
    PathsAndConsumesOfModules                     pathsAndConsumesOfModules_;
    edm::propagate_const<std::unique_ptr<Schedule>> schedule_;
    std::vector<SubProcess> subProcesses_;
    edm::propagate_const<std::unique_ptr<HistoryAppender>> historyAppender_;

    edm::propagate_const<std::unique_ptr<FileBlock>> fb_;
    edm::propagate_const<std::shared_ptr<EDLooperBase>> looper_;

    //The atomic protects concurrent access of deferredExceptionPtr_
    std::atomic<bool>                             deferredExceptionPtrIsSet_;
    std::exception_ptr                            deferredExceptionPtr_;
    
    SharedResourcesAcquirer                       sourceResourcesAcquirer_;
    std::shared_ptr<std::recursive_mutex>         sourceMutex_;
    PrincipalCache                                principalCache_;
    bool                                          beginJobCalled_;
    bool                                          shouldWeStop_;
    bool                                          stateMachineWasInErrorState_;
    std::string                                   fileMode_;
    std::string                                   emptyRunLumiMode_;
    std::string                                   exceptionMessageFiles_;
    std::string                                   exceptionMessageRuns_;
    std::string                                   exceptionMessageLumis_;
    bool                                          alreadyHandlingException_;
    bool                                          forceLooperToEnd_;
    bool                                          looperBeginJobRun_;
    bool                                          forceESCacheClearOnNewRun_;

    int                                           numberOfForkedChildren_;
    unsigned int                                  numberOfSequentialEventsPerChild_;
    bool                                          setCpuAffinity_;
    bool                                          continueAfterChildFailure_;
    
    PreallocationConfiguration                    preallocations_;
    
    bool                                          asyncStopRequestedWhileProcessingEvents_;
    InputSource::ItemType                         nextItemTypeFromProcessingEvents_;
    StatusCode                                    asyncStopStatusCodeFromProcessingEvents_;
    bool firstEventInBlock_=true;
    
    typedef std::set<std::pair<std::string, std::string> > ExcludedData;
    typedef std::map<std::string, ExcludedData> ExcludedDataMap;
    ExcludedDataMap                               eventSetupDataToExcludeFromPrefetching_;
    
    bool printDependencies_ = false;
  }; // class EventProcessor

  //--------------------------------------------------------------------

  inline
  EventProcessor::StatusCode
  EventProcessor::run() {
    return runToCompletion();
  }
}
#endif
