#ifndef Framework_RunStopwatch_h
#define Framework_RunStopwatch_h

/*----------------------------------------------------------------------
  
$Id: RunStopwatch.h,v 1.0 2006/02/08 00:44:25 wmtan Exp $

Simple "guard" class as suggested by Chris Jones to start/stop the
Stopwatch: creating an object of type RunStopwatch starts the clock
pointed to, deleting it (when it goes out of scope) automatically
calls the destructor which stops the clock.

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"
#include "boost/scoped_ptr.hpp"
#include "TStopwatch.h"

namespace edm {

  class RunStopwatch {

  public:
    typedef boost::shared_ptr<TStopwatch> StopwatchPointer;

    RunStopwatch(const StopwatchPointer& ptr): stopwatch_(ptr) {
      stopwatch_->Start(false);
    }

    ~RunStopwatch(){
      stopwatch_->Stop();
    }

  private:
    StopwatchPointer stopwatch_;

  };

}
#endif
