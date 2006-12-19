#ifndef Framework_EventProcessor_h
#define Framework_EventProcessor_h

/*----------------------------------------------------------------------

EventProcessor: This defines the 'framework application' object. It is
configured in the user's main() function, and is set running.

Requirements placed upon the command line processor:

 1. Support registering of switches/options from user modules
    a. make help available
    b. make sure there are no collisions.

 2. If a switch or option is not supplied, the module looking for it
    must have a sensible default behavior. There should be no required
    switches, nor required options.

Software policing seems needed in order to provent illicit use to
configure moddules entirely with passed arguments, rather than using
the ParameterSet which the module is passed at the time of its
creation.


problems:
  specification of "pass" and other things like it - things that
  have to do with the job as a whole or with this object in particular.

  we are not careful yet about catching seal exceptions and printing
  useful information.

  where does the pluginmanager initialise call go?


$Id: EventProcessor.h,v 1.28 2006/11/17 00:35:27 wmtan Exp $

----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "sigc++/signal.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/Actions.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/PassID.h"
#include "DataFormats/Common/interface/ReleaseVersion.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventHelperDescription.h"

namespace edm {
class EDLooper;
class EDLooperHelper;

  namespace event_processor
  {  
    /*
      ------------
      cause events to be processed in a separate thread
      and functions used in the online.  Several of these
      state are likely to be transitory in the offline
      because they are completly driven by the data coming
      from the input source.
    */
    enum State { sInit=0,sJobReady,sRunGiven,sRunning,sStopping,
		 sShuttingDown,sDone,sJobEnded,sError,sErrorEnded,sEnd,sInvalid };
    
    enum Msg { mSetRun=0, mSkip, mRunAsync, mRunID, mRunCount, mBeginJob,
	       mStopAsync, mShutdownAsync, mEndJob, mCountComplete,
	       mInputExhausted, mStopSignal, mShutdownSignal, mFinished,
	       mAny, mDtor, mException, mInputRewind };

    class StateSentry;
  }
    
  class EventProcessor
  {
    // ------------ friend classes and functions ----------------
    friend class edm::EDLooperHelper;

  public:

    // Status codes:
    //   0     successful completion
    //   1     exception of unknown type caught
    //   2     everything else
    //   3     signal received
    //   4     input complete
    enum Status { epSuccess=0, epException=1, epOther=2, epSignal=3,
    epInputComplete=4 };

    // Eventually, we might replace StatusCode with a class. This
    // class should have an automatic conversion to 'int'.
    typedef Status StatusCode ;


    /// The input string contains the entire contents of a
    /// configuration file. Uses default constructed ServiceToken, so
    /// an EventProcessor created with this constructor will allow
    /// access to no externally-created services.
    /// This should become pretty much useless when construction of
    /// services is moved outside of the EventProcessor.
    /// explicit EventProcessor(const std::string& config);


    // The input string 'config' contains the entire contents of a  configuration file.
    // Also allows the attachement of pre-existing services specified  by 'token', and
    // the specification of services by name only (defaultServices and forcedServices).
    // 'defaultServices' are overridden by 'config'.
    // 'forcedServices' cause an exception if the same service is specified in 'config'.
    explicit EventProcessor(std::string const& config,
			    ServiceToken const& token = ServiceToken(),
			    serviceregistry::ServiceLegacy =
			      serviceregistry::kOverlapIsError,
			    std::vector<std::string> const& defaultServices =
			      std::vector<std::string>(),
			    std::vector<std::string> const& forcedServices =
			      std::vector<std::string>());

    // Same as previous constructor, but without a 'token'.  Token will be defaulted.

    explicit EventProcessor(std::string const& config,
			    std::vector<std::string> const& defaultServices,
			    std::vector<std::string> const& forcedServices =
			    std::vector<std::string>());
    


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


    char const* currentStateName() const;
    char const* stateName(event_processor::State s) const;
    char const* msgName(event_processor::Msg m) const;
    event_processor::State getState() const;
    void runAsync();
    StatusCode statusAsync() const;
    StatusCode stopAsync(); // wait for the completion
    StatusCode shutdownAsync(); // wait for the completion
    StatusCode waitTillDoneAsync(); // wait until InputExhausted
    void setRunNumber(RunNumber_t runNumber);

    // -------------

    // Invoke event processing.  We will process a total of
    // 'numberToProcess' events. If numberToProcess is zero, we will
    // process events intil the input sources are exhausted. If it is
    // given a non-zero number, processing continues until either (1)
    // this number of events has been processed, or (2) the input
    // sources are exhausted.
    StatusCode run(unsigned int numberToProcess);

    // Process until the input source is exhausted.
    StatusCode run();

    // Process one event with the given EventID
    StatusCode run(EventID const& id);

    // Skip the specified number of events, and then process the next event.
    // If numberToSkip is negative, we will back up.
    // For example, skip(-1) processes the previous event.
    StatusCode skip(int numberToSkip);

    InputSource& getInputSource();

    /// Get the main input source, dynamic_casting it to type T. This
    /// will throw an execption if an inappropriate type T is used.
    template <class T> 
    T& 
    getSpecificInputSource();


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

    //------------------------------------------------------------------
    //
    // Data members (and nested classes and structs) below.

    /// signal is emitted after the Event has been created by the
    /// InputSource but before any modules have seen the Event
    ActivityRegistry::PreProcessEvent
    preProcessEventSignal;

    /// signal is emitted after all modules have finished processing
    /// the Event
    ActivityRegistry::PostProcessEvent
    postProcessEventSignal;


    struct CommonParams
    {
      CommonParams():
	processName_(),
	releaseVersion_(),
	passID_()
      { }

      CommonParams(std::string const& processName,
		   ReleaseVersion const& releaseVersion,
		   PassID const& passID):
	processName_(processName),
	releaseVersion_(releaseVersion),
	passID_(passID)
      { }
      
      std::string processName_;
      ReleaseVersion releaseVersion_;
      PassID passID_;
    }; // struct CommonParams

    // Really should not be public,
    //   but the EventFilter needs it for now.    
    ServiceToken getToken();

  private:
    // init() is used by only by constructors
    void init(std::string const& config,
		ServiceToken const& token,
		serviceregistry::ServiceLegacy,
		std::vector<std::string> const& defaultServices,
		std::vector<std::string> const& forcedServices);
  
    StatusCode run_p(unsigned int numberToProcess,
		     event_processor::Msg m);
    StatusCode doneAsync(event_processor::Msg m);
    EventHelperDescription runOnce();
    
    void rewind();

    void connectSigs(EventProcessor* ep);

    struct DoPluginInit
    {
      DoPluginInit();
    };

    void changeState(event_processor::Msg);
    void errorState();
    void setupSignal();

    static void asyncRun(EventProcessor*);

    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.    


    DoPluginInit                                  plug_init_;
    CommonParams                                  common_;
    boost::shared_ptr<ActivityRegistry>           actReg_;
    WorkerRegistry                                wreg_;
    SignallingProductRegistry                     preg_;
    ServiceToken                                  serviceToken_;
    boost::shared_ptr<InputSource>                input_;
    std::auto_ptr<Schedule>                       schedule_;
    std::auto_ptr<eventsetup::EventSetupProvider> esp_;    
    ActionTable                                   act_table_;

    volatile event_processor::State               state_;
    boost::shared_ptr<boost::thread>              event_loop_;

    boost::mutex                                  state_lock_;
    boost::mutex                                  stop_lock_;
    boost::condition                              stopper_;
    volatile int                                  stop_count_;
    volatile Status                               last_rc_;
    std::string                                   last_error_text_;
    volatile bool                                 id_set_;
    volatile pthread_t                            event_loop_id_;
    int                                           my_sig_num_;
    boost::shared_ptr<edm::EDLooper>              looper_;

    friend class event_processor::StateSentry;
  }; // class EventProcessor

  //--------------------------------------------------------------------
  // ----- implementation details below ------
  
  inline
  EventProcessor::StatusCode
  EventProcessor::run() {
    return run(0);
  }

  template <class T> T& EventProcessor::getSpecificInputSource()
  {
    InputSource& is = this->getInputSource();
    return dynamic_cast<T&>(is);
  }

}
#endif
