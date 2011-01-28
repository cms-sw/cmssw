#ifndef FWCore_Framework_EventProcessor_h
#define FWCore_Framework_EventProcessor_h

/*----------------------------------------------------------------------

EventProcessor: This defines the 'framework application' object. It is
configured in the user's main() function, and is set running.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/condition.hpp"
#include "boost/utility.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace statemachine {
  class Machine;
  class Run;
}

namespace edm {

  class ActionTable;
  class EDLooperBase;
  class ProcessDesc;
  class SubProcess;
  namespace eventsetup {
    class EventSetupProvider;
  }

  namespace event_processor {
    /*
      Several of these state are likely to be transitory in
      the offline because they are completly driven by the
      data coming from the input source.
    */
    enum State { sInit = 0, sJobReady, sRunGiven, sRunning, sStopping,
                 sShuttingDown, sDone, sJobEnded, sError, sErrorEnded, sEnd, sInvalid };

    enum Msg { mSetRun = 0, mSkip, mRunAsync, mRunID, mRunCount, mBeginJob,
               mStopAsync, mShutdownAsync, mEndJob, mCountComplete,
               mInputExhausted, mStopSignal, mShutdownSignal, mFinished,
               mAny, mDtor, mException, mInputRewind };

    class StateSentry;
  }

  class EventProcessor : public IEventProcessor, private boost::noncopyable {
  public:

    // The input string 'config' contains the entire contents of a  configuration file.
    // Also allows the attachement of pre-existing services specified  by 'token', and
    // the specification of services by name only (defaultServices and forcedServices).
    // 'defaultServices' are overridden by 'config'.
    // 'forcedServices' cause an exception if the same service is specified in 'config'.
    explicit EventProcessor(std::string const& config,
                            ServiceToken const& token = ServiceToken(),
                            serviceregistry::ServiceLegacy = serviceregistry::kOverlapIsError,
                            std::vector<std::string> const& defaultServices = std::vector<std::string>(),
                            std::vector<std::string> const& forcedServices = std::vector<std::string>());

    // Same as previous constructor, but without a 'token'.  Token will be defaulted.

    EventProcessor(std::string const& config,
                   std::vector<std::string> const& defaultServices,
                   std::vector<std::string> const& forcedServices = std::vector<std::string>());

    EventProcessor(boost::shared_ptr<ProcessDesc>& processDesc,
                   ServiceToken const& token,
                   serviceregistry::ServiceLegacy legacy);

    /// meant for unit tests
    EventProcessor(std::string const& config, bool isPython);

    ~EventProcessor();

    /**This should be called before the first call to 'run'
       If this is not called in time, it will automatically be called
       the first time 'run' is called
       */
    void beginJob();

    /**This should be called before the EventProcessor is destroyed
       throws if any module's endJob throws an exception.
       */
    void endJob();

    /**Member functions to support asynchronous interface.
       */

    char const* currentStateName() const;
    char const* stateName(event_processor::State s) const;
    char const* msgName(event_processor::Msg m) const;
    event_processor::State getState() const;
    void runAsync();
    StatusCode statusAsync() const;

    // Concerning the async control functions:
    // The event processor is left with the running thread.
    // The async thread is stuck at this point and the process
    // is likely not going to be able to continue.
    // The reason for this timeout could be either an infinite loop
    // or I/O blocking forever.
    // The only thing to do is end the process.
    // If you call endJob, you will likely get an exception from the
    // state checks telling you that it is not valid to call this function.
    // All these function force the event processor state into an
    // error state.

    // tell the event loop to stop and wait for its completion
    StatusCode stopAsync(unsigned int timeout_secs = 60 * 2);

    // tell the event loop to shutdown and wait for the completion
    StatusCode shutdownAsync(unsigned int timeout_secs = 60 * 2);

    // wait until async event loop thread completes
    // or timeout occurs (See StatusCode for return values)
    StatusCode waitTillDoneAsync(unsigned int timeout_seconds = 0);

    // Both of these calls move the EP to the ready to run state but only
    // the first actually sets the run number, the other one just stores
    // the run number set externally in order to later compare to the one
    // read from the input source for verification
    void setRunNumber(RunNumber_t runNumber);
    void declareRunNumber(RunNumber_t runNumber);

    // -------------

    // These next two functions are deprecated.  Please use
    // RunToCompletion or RunEventCount instead.  These will
    // be deleted as soon as we have time to clean up the code
    // in packages outside the Framework that uses them already.
    StatusCode run(int numberEventsToProcess, bool repeatable = true);
    StatusCode run();

    // Skip the specified number of events.
    // If numberToSkip is negative, we will back up.
    StatusCode skip(int numberToSkip);

    // Rewind to the first event
    void rewind();

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

    /// signal is emitted after the Event has been created by the
    /// InputSource but before any modules have seen the Event
    ActivityRegistry::PreProcessEvent&
    preProcessEventSignal() {return preProcessEventSignal_;}

    /// signal is emitted after all modules have finished processing
    /// the Event
    ActivityRegistry::PostProcessEvent&
    postProcessEventSignal() {return postProcessEventSignal_;}

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
    // The function "runEventCount" will pause after processing the
    // number of input events specified by the argument.  One can
    // call it again to resume processing at the same point.  This
    // function will also stop at the same point as "runToCompletion"
    // if the job is complete before the requested number of events
    // are processed.  If the requested number of events is less than
    // 1, "runEventCount" interprets this as infinity and does not
    // pause until the job is complete.
    //
    // The return values from these functions are as follows:
    //   epSignal - processing terminated early, SIGUSR2 encountered
    //   epCountComplete - "runEventCount" processed the number of events
    //                     requested by the argument
    //   epSuccess - all other cases
    //
    // We expect that in most cases, processes will call
    // "runToCompletion" once per job and not use "runEventCount".
    //
    // If a process used "runEventCount", then it would need to
    // check the value returned by "runEventCount" to determine
    // if it processed the requested number of events.  It would
    // only make sense to call it again if it returned epCountComplete
    // on the preceding call.

    // The online is an exceptional case.  Online uses the DaqSource
    // and the StreamerOutputModule, which are specially written to
    // handle multiple calls of "runToCompletion" in the same job.
    // The call to setRunNumber resets the DaqSource between those calls.
    // With most sources and output modules, this does not work.
    // If and only if called by the online, the argument to runToCompletion
    // is set to true and this affects the state initial and final state
    // transitions that are managed directly in EventProcessor.cc. (I am
    // not sure if there is a reason for this or it is just a historical
    // peculiarity that could be cleaned up and removed).

    virtual StatusCode runToCompletion(bool onlineStateTransitions);
    virtual StatusCode runEventCount(int numberOfEventsToProcess);

    // The following functions are used by the code implementing our
    // boost statemachine

    virtual void readFile();
    virtual void closeInputFile();
    virtual void openOutputFiles();
    virtual void closeOutputFiles();

    virtual void respondToOpenInputFile();
    virtual void respondToCloseInputFile();
    virtual void respondToOpenOutputFiles();
    virtual void respondToCloseOutputFiles();

    virtual void startingNewLoop();
    virtual bool endOfLoop();
    virtual void rewindInput();
    virtual void prepareForNextLoop();
    virtual bool shouldWeCloseOutput() const;

    virtual void doErrorStuff();

    virtual void beginRun(statemachine::Run const& run);
    virtual void endRun(statemachine::Run const& run);

    virtual void beginLumi(ProcessHistoryID const& phid, int run, int lumi);
    virtual void endLumi(ProcessHistoryID const& phid, int run, int lumi);

    virtual statemachine::Run readAndCacheRun();
    virtual int readAndCacheLumi();
    virtual void writeRun(statemachine::Run const& run);
    virtual void deleteRunFromCache(statemachine::Run const& run);
    virtual void writeLumi(ProcessHistoryID const& phid, int run, int lumi);
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, int run, int lumi);

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
    void init(boost::shared_ptr<ProcessDesc>& processDesc,
              ServiceToken const& token,
              serviceregistry::ServiceLegacy);

    StatusCode runCommon(bool onlineStateTransitions, int numberOfEventsToProcess);
    void terminateMachine();

    StatusCode doneAsync(event_processor::Msg m);

    StatusCode waitForAsyncCompletion(unsigned int timeout_seconds);

    void connectSigs(EventProcessor* ep);

    void changeState(event_processor::Msg);
    void errorState();
    void setupSignal();

    static void asyncRun(EventProcessor*);

    bool hasSubProcess() const {
      return subProcess_.get() != 0;
    }

    //------------------------------------------------------------------
    //
    // Data members below.
    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.

    ActivityRegistry::PreProcessEvent             preProcessEventSignal_;
    ActivityRegistry::PostProcessEvent            postProcessEventSignal_;
    boost::shared_ptr<ActivityRegistry>           actReg_;
    boost::shared_ptr<SignallingProductRegistry>  preg_;
    ServiceToken                                  serviceToken_;
    boost::shared_ptr<InputSource>                input_;
    boost::shared_ptr<eventsetup::EventSetupProvider> esp_;
    boost::shared_ptr<ActionTable const>          act_table_;
    boost::shared_ptr<ProcessConfiguration>       processConfiguration_;
    std::auto_ptr<Schedule>                       schedule_;
    std::auto_ptr<SubProcess>                     subProcess_;

    volatile event_processor::State               state_;
    boost::shared_ptr<boost::thread>              event_loop_;

    boost::mutex                                  state_lock_;
    boost::mutex                                  stop_lock_;
    boost::condition                              stopper_;
    boost::condition                              starter_;
    volatile int                                  stop_count_;
    volatile Status                               last_rc_;
    std::string                                   last_error_text_;
    volatile bool                                 id_set_;
    volatile pthread_t                            event_loop_id_;
    int                                           my_sig_num_;
    boost::shared_ptr<FileBlock>                  fb_;
    boost::shared_ptr<EDLooperBase>               looper_;

    std::auto_ptr<statemachine::Machine>          machine_;
    PrincipalCache                                principalCache_;
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
    typedef std::set<std::pair<std::string, std::string> > ExcludedData;
    typedef std::map<std::string, ExcludedData> ExcludedDataMap;
    ExcludedDataMap                               eventSetupDataToExcludeFromPrefetching_;
    friend class event_processor::StateSentry;
  }; // class EventProcessor

  //--------------------------------------------------------------------

  inline
  EventProcessor::StatusCode
  EventProcessor::run() {
    return run(-1, false);
  }
}
#endif
