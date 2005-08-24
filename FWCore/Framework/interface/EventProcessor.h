#ifndef EDM_EVENTPROC_H
#define EDM_EVENTPROC_H

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

The simplest 'main' that uses this should look like:

#include "FWCore/Framework/interface/EventProcessor.h"
int main(int argc, char* argv[])
{
  edm::EventProcessor app(argc, argv);
  return app.run();
}

CAUTION: The phase 1 release does not deal gracefully with exceptions
thrown by the constructor, for this simple version of main.
More friendly is:

#include <iostream>
#include "FWCore/Framework/interface/EventProcessor.h"
int main(int argc, char* argv[])
{
  try 
  {
    edm::EventProcessor app(argc, argv);
    return app.run();
  }
  catch (...)
  {
    std::cerr << "Failed to create framework object\n";
  }
}

More sophisticated error handling is also possible.

problems:
  specification of "pass" and other things like it - things that
  have to do with the job as a whole or with this object in particular.

  we are not careful yet about catching seal exceptions and printing
  useful information.

  where does the pluginmanager initialise call go?


$Id: EventProcessor.h,v 1.3 2005/07/14 22:50:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <string>
#include "boost/signal.hpp"

namespace edm {

  class FwkImpl;
  class Event;
  class EventSetup;

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

    // This accepts the argc, argv from main(). Only 'switches' and
    // 'options' are expected. 
    EventProcessor(int argc, char* argv[]);

    // The input string contains the entire contents of a
    // configuration file.    
    explicit EventProcessor(const std::string& config);

    // This constructor combines the effect of the two above.
    EventProcessor(int argc, char* argv[], const std::string& config);

    ~EventProcessor();

    // Invoke event processing.  We will process a total of
    // 'numberToProcess' events. If numberToProcess is zero, we will
    // process events intil the input sources are exhausted. If it is
    // given a non-zero number, processing continues until either (1)
    // this number of events has been processed, or (2) the input
    // sources are exhausted.
    StatusCode run(unsigned long numberToProcess = 0);

    boost::signal<void (const Event&, const EventSetup&)> preProcessEventSignal;
    boost::signal<void (const Event&, const EventSetup&)> postProcessEventSignal;
    
  private:
    FwkImpl* impl_;
  };
  
}
#endif
