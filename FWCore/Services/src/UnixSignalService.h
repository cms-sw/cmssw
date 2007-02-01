#ifndef Services_UnixSignalService_h
#define Services_UnixSignalService_h

/*----------------------------------------------------------------------

UnixSignalService: At present, this defines a SIGUSR2 handler and
sets the shutdown flag when that signal has been raised.

This service is instantiated at job startup.

----------------------------------------------------------------------*/

#include <string>
#include <signal.h>

#include "sigc++/signal.h"
#include "boost/thread/thread.hpp"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class Event;
  class EventSetup;

  namespace service {

  class UnixSignalService
  {

  public:
    UnixSignalService(edm::ParameterSet const& ps, edm::ActivityRegistry& ac); 
    ~UnixSignalService();
//  void postEventProcessing( const Event& ev, const EventSetup& es );

  }; // class UnixSignalService
  }  // end of namespace service
}    // end of namespace edm
#endif
