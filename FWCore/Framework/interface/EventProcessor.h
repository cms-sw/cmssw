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
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/condition.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <mutex>
#include <exception>

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
    virtual StatusCode runToCompletion();

    // The following functions are used by the code implementing our
    // boost statemachine

    virtual void readFile();
    virtual void closeInputFile(bool cleaningUpAfterException);
    virtual void openOutputFiles();
    virtual void closeOutputFiles();

    virtual void respondToOpenInputFile();
    virtual void respondToCloseInputFile();

    virtual void startingNewLoop();
    virtual bool endOfLoop();
    virtual void rewindInput();
    virtual void prepareForNextLoop();
    virtual bool shouldWeCloseOutput() const;

    virtual void doErrorStuff();

    virtual void beginRun(statemachine::Run const& run);
    virtual void endRun(statemachine::Run const& run, bool cleaningUpAfterException);

    virtual void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);
    virtual void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException);

    virtual statemachine::Run readRun();
    virtual statemachine::Run readAndMergeRun();
    virtual int readLuminosityBlock();
    virtual int readAndMergeLumi();
    virtual void writeRun(statemachine::Run const& run);
    virtual void deleteRunFromCache(statemachine::Run const& run);
    virtual void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);

    virtual void readAndProcessEvent();
    virtual bool shouldWeStop() const;

    virtual void setExceptionMessageFiles(std::string& message);
    virtual void setExceptionMessageRuns(std::string& message);
    virtual void setExceptionMessageLumis(std::string& message);

    virtual bool alreadyHandlingException() const;

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

    void terminateMachine(std::auto_ptr<statemachine::Machine>&);
    std::auto_ptr<statemachine::Machine> createStateMachine();

    void setupSignal();

    bool hasSubProcess() const {
      return subProcess_.get() != 0;
    }

    void possiblyContinueAfterForkChildFailure();
    
    friend class StreamProcessingTask;
    void processEventsForStreamAsync(unsigned int iStreamIndex,
                                     std::atomic<bool>* finishedProcessingEvents);
    
    
    //read the next event using Stream iStreamIndex
    void readEvent(unsigned int iStreamIndex);

    //process the already read event using Stream iStreamIndex
    void processEvent(unsigned int iStreamIndex);

    //returns true if an asynchronous stop was requested
    bool checkForAsyncStopRequest(StatusCode&);
    //------------------------------------------------------------------
    //
    // Data members below.
    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.

    std::shared_ptr<ActivityRegistry>           actReg_;
    std::shared_ptr<ProductRegistry const>      preg_;
    std::shared_ptr<BranchIDListHelper>         branchIDListHelper_;
    std::shared_ptr<ThinnedAssociationsHelper>  thinnedAssociationsHelper_;
    ServiceToken                                  serviceToken_;
    std::unique_ptr<InputSource>                  input_;
    std::unique_ptr<eventsetup::EventSetupsController> espController_;
    boost::shared_ptr<eventsetup::EventSetupProvider> esp_;
    std::unique_ptr<ExceptionToActionTable const>          act_table_;
    std::shared_ptr<ProcessConfiguration const>       processConfiguration_;
    ProcessContext                                processContext_;
    PathsAndConsumesOfModules                     pathsAndConsumesOfModules_;
    std::auto_ptr<Schedule>                       schedule_;
    std::auto_ptr<SubProcess>                     subProcess_;
    std::unique_ptr<HistoryAppender>            historyAppender_;

    std::unique_ptr<FileBlock>                    fb_;
    boost::shared_ptr<EDLooperBase>               looper_;

    //The atomic protects concurrent access of deferredExceptionPtr_
    std::atomic<bool>                             deferredExceptionPtrIsSet_;
    std::exception_ptr                            deferredExceptionPtr_;
    
    std::mutex                                    nextTransitionMutex_;
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
    
    typedef std::set<std::pair<std::string, std::string> > ExcludedData;
    typedef std::map<std::string, ExcludedData> ExcludedDataMap;
    ExcludedDataMap                               eventSetupDataToExcludeFromPrefetching_;
  }; // class EventProcessor

  //--------------------------------------------------------------------

  inline
  EventProcessor::StatusCode
  EventProcessor::run() {
    return runToCompletion();
  }
}
#endif
