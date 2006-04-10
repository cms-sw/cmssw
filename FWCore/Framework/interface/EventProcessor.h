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


$Id: EventProcessor.h,v 1.14 2006/03/15 21:24:49 wmtan Exp $

----------------------------------------------------------------------*/

#include <string>
#include "boost/signal.hpp"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"

namespace edm {

  class FwkImpl;
  class Event;
  class EventSetup;
  class EventID;
  class Timestamp;
  class ServiceToken;
  class InputSource;
  
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


    /// The input string contains the entire contents of a
    /// configuration file. Uses default constructed ServiceToken, so
    /// an EventProcessor created with this constructor will allow
    /// access to no externally-created services.
    /// This should become pretty much useless when construction of
    /// services is moved outside of the EventProcessor.
    explicit EventProcessor(const std::string& config);


    // The input string contains the entire contents of a
    // configuration file.  Same as previous constructor, except allow
    // attachement of pre-existing services.
    EventProcessor(const std::string& config,
		   const ServiceToken&,
		   serviceregistry::ServiceLegacy);

    ~EventProcessor();

    /**This should be called before the first call to 'run'
       If this is not called in time, it will automatically be called
       the first time 'run' is called
       */
    void beginJob();

    /**This should be called before the EventProcessor is destroyed
       returns false if any module's endJob throws an exception
       */
    bool endJob();

    /*
      ------------
      cause events to be processed in a separate thread
      and functions used in the online.  Several of these
      state are likely to be tranitory in the offline
      because they are completly driven by the data coming from the 
      input source.

      sInit: ctor has completed
      sJobStart: beginJob is active or complete
      sRunStart: beginRun is active or complete
      sRunning: event loop is actively processing events
      sStopping: event loop is supposed to shut down after the current event
      sIdle: no event loop is active
      sError: event loop has encountered a bad error and stopped
      sRunEnd: endRun is active or complete
      sJobEnd: endJob is active or complete
    */
    enum State {sInit,sJobStart,sRunStart,sRunning,sStopping,
		sIdle,sError,sRunEnd,sJobEnd };

    State getState() const;
    void runAsync();
    StatusCode statusAsync() const;
    StatusCode stopAsync(); // wait for the completion
    void beginRun();
    void endRun();
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
    
  private:
    FwkImpl* impl_;
  };
  
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
