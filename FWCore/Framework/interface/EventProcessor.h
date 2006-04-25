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


$Id: EventProcessor.h,v 1.16 2006/04/19 21:03:48 jbk Exp $

----------------------------------------------------------------------*/

#include <string>

#include "boost/signal.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/Actions.h"
#include "DataFormats/Common/interface/EventID.h"

namespace edm {

  class Event;
  class EventSetup;
  class EventID;
  class Timestamp;
  class InputSource;
  class ActivityRegistry;
  class Schedule;

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
	       mAny, mDtor, mException };

    class StateSentry;
  }
    
  class EventProcessor
  {
  public:

    // Eventually, we might replace StatusCode with a class. This
    // class should have an automatic conversion to 'int'.
    typedef int StatusCode ;

    // Status codes:
    //   0     successful completion
    //   1     exception of unknown type caught
    //   2     everything else
    //   3     signal received
    //   4     input complete
    enum Status { epSuccess=0, epException=1, epOther=2, epSignal=3,
    epInputComplete=4 };


    /// The input string contains the entire contents of a
    /// configuration file. Uses default constructed ServiceToken, so
    /// an EventProcessor created with this constructor will allow
    /// access to no externally-created services.
    /// This should become pretty much useless when construction of
    /// services is moved outside of the EventProcessor.
    /// explicit EventProcessor(const std::string& config);


    // The input string contains the entire contents of a
    // configuration file.  Same as previous constructor, except allow
    // attachement of pre-existing services.
    explicit EventProcessor(const std::string& config,
			    const ServiceToken& = ServiceToken(),
			    serviceregistry::ServiceLegacy =
			    serviceregistry::kOverlapIsError);

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


    const char* currentStateName() const;
    const char* stateName(event_processor::State s) const;
    const char* msgName(event_processor::Msg m) const;
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
    StatusCode run(unsigned long numberToProcess);

    // Process until the input source is exhausted.
    StatusCode run();

    // Process one event with the given EventID
    StatusCode run(const EventID& id);

    // Skip the specified number of events, and then process the next event.
    // If numberToSkip is negative, we will back up.
    // For example, skip(-1) processes the previous event.
    StatusCode skip(long numberToSkip);

    InputSource& getInputSource();

    /// Get the main input source, dynamic_casting it to type T. This
    /// will throw an execption if an inappropriate type T is used.
    template <class T> 
    T& 
    getSpecificInputSource();

    /// signal is emitted after the Event has been created by the
    /// InputSource but before any modules have seen the Event
    boost::signal<void (const EventID&, const Timestamp&)> 
    preProcessEventSignal;

    /// signal is emitted after all modules have finished processing
    /// the Event
    boost::signal<void (const Event&, const EventSetup&)> 
    postProcessEventSignal;
    
    struct CommonParams
    {
      CommonParams():
	version_(),
	pass_()
      { }

      CommonParams(const std::string& name,
		   unsigned long ver,
		   unsigned long pass):
	processName_(name),
	version_(ver),
	pass_(pass)
      { }
      
      std::string             processName_;
      unsigned long           version_;
      unsigned long           pass_;
    }; // struct CommonParams

  private:

    StatusCode run_p(unsigned long numberToProcess,
		     event_processor::Msg m);
    StatusCode doneAsync(event_processor::Msg m);

    ServiceToken   getToken();
    void           connectSigs(EventProcessor* ep);

    struct DoPluginInit
    {
      DoPluginInit();
    };

    // Are all these data members really needed? Some of them are used
    // only during construction, and never again. If they aren't
    // really needed, we should remove them.    
    //shared_ptr<ParameterSet>        params_;

    DoPluginInit plug_init_;
    CommonParams common_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    WorkerRegistry wreg_;
    SignallingProductRegistry preg_;
    ServiceToken serviceToken_;
    boost::shared_ptr<InputSource> input_;
    std::auto_ptr<Schedule> sched_;
    std::auto_ptr<eventsetup::EventSetupProvider> esp_;    
    ActionTable act_table_;
    volatile event_processor::State state_;

    void changeState(event_processor::Msg);
    void errorState();
    void setupSignal();

    static void asyncRun(EventProcessor*);
    boost::shared_ptr<boost::thread> event_loop_;

    boost::mutex state_lock_;
    boost::mutex stop_lock_;
    boost::condition stopper_;
    volatile int stop_count_;
    volatile Status last_rc_;
    std::string last_error_text_;
    volatile bool id_set_;
    volatile pthread_t event_loop_id_;
    int my_sig_num_;

    friend class event_processor::StateSentry;
  };

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
