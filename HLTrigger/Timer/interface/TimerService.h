#ifndef Timer_Service_
#define Timer_Service_

/**\class TimerService

Description: Class accessing CPUTimer to record processing-time info per module
(either CPU-time or wall-clock-time)

Original Author:  Christos Leonidopoulos, March 2007

*/

#include "sigc++/signal.h"

#include "FWCore/Utilities/interface/CPUTimer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>

#ifdef __linux
#include <time.h>
#else
typedef int clockid_t;
#define CLOCK_REALTIME               0                                                                                                                                                                                        
#define CLOCK_MONOTONIC              1                                                                                                                                                                                        
#define CLOCK_PROCESS_CPUTIME_ID     2                                                                                                                                                                                        
#define CLOCK_THREAD_CPUTIME_ID      3                                                                                                                                                                                        
#endif

namespace hlt {

  class CPUTimer {
  public:
    explicit CPUTimer(bool cpu = true) :
      timer_( cpu ? CLOCK_THREAD_CPUTIME_ID : CLOCK_REALTIME )
    {
      reset();
    }

    void reset() {
      start_.tv_sec  = 0;
      start_.tv_nsec = 0;
      stop_.tv_sec   = 0;
      stop_.tv_nsec  = 0;
    }

    void start() {
#ifdef __linux
      clock_gettime(timer_, & start_);
#endif
    }

    void stop() {
#ifdef __linux
      clock_gettime(timer_, & stop_);
#endif
    }

  // return the delta between start and stop in seconds
  double delta() const {
    if (stop_.tv_nsec > start_.tv_nsec)
      return (double) (stop_.tv_sec - start_.tv_sec) + (double) (stop_.tv_nsec - start_.tv_nsec) / (double) 1e9;
    else
      return (double) (stop_.tv_sec - start_.tv_sec) - (double) (start_.tv_nsec - stop_.tv_nsec) / (double) 1e9;
  }

  private:
    const clockid_t timer_;
    timespec        start_; 
    timespec        stop_;
  };

} // namespace hlt


class TimerService {
 public:
  TimerService(const edm::ParameterSet&, edm::ActivityRegistry& iAR);
  ~TimerService();

  // signal with module-description and processing time (in secs)
  sigc::signal<void, const edm::ModuleDescription&, double> newMeasurementSignal;

  // fwk calls this method before a module is processed
  void preModule(const edm::ModuleDescription& iMod);
  // fwk calls this method after a module has been processed
  void postModule(const edm::ModuleDescription& iMod);

 private:
  // whether to use CPU-time (default) or wall-clock time
  bool useCPUtime;

  // cpu-timer
  hlt::CPUTimer cpu_timer;

  // true is the process is bound to a single CPU
  bool is_bound_;
};

#endif // #define Timer_Service_
