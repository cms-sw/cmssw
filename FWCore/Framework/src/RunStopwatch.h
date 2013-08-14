#ifndef FWCore_Framework_RunStopwatch_h
#define FWCore_Framework_RunStopwatch_h

/*----------------------------------------------------------------------
  

Simple "guard" class as suggested by Chris Jones to start/stop the
Stopwatch: creating an object of type RunStopwatch starts the clock
pointed to, deleting it (when it goes out of scope) automatically
calls the destructor which stops the clock.

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"
#include "FWCore/Utilities/interface/CPUTimer.h"

namespace edm {

  class RunStopwatch {

  public:
    typedef boost::shared_ptr<CPUTimer> StopwatchPointer;

    RunStopwatch(const StopwatchPointer& ptr): stopwatch_(ptr) {
      if(stopwatch_) {
        stopwatch_->start();
      }
    }

    ~RunStopwatch(){
      if(stopwatch_) {
        stopwatch_->stop();
      }
    }

  private:
    StopwatchPointer stopwatch_;

  };

  class RunDualStopwatches {
    
  public:
    typedef boost::shared_ptr<CPUTimer> StopwatchPointer;
    
    RunDualStopwatches(const StopwatchPointer& ptr1, CPUTimer* const ptr2): stopwatch1_(ptr1),stopwatch2_(ptr2) {
      if(stopwatch1_ && 0 != stopwatch2_) {
        stopwatch1_->start();
      }
    }
    
    ~RunDualStopwatches(){
      if (stopwatch1_ && 0 != stopwatch2_) {
        stopwatch2_->add(stopwatch1_->stop());
      }
    }
    
  private:
    StopwatchPointer stopwatch1_;
    CPUTimer* const stopwatch2_;
    
  };
  
}
#endif
