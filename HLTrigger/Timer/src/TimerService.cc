#include <iostream>
#include <sched.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/Timer/interface/TimerService.h"
#include "HLTrigger/Timer/interface/CPUAffinity.h"

TimerService::TimerService(const edm::ParameterSet& ps, 
                                 edm::ActivityRegistry& iAR) :
  useCPUtime( ps.getUntrackedParameter<bool>("useCPUtime", true) ),
  cpu_timer(useCPUtime),
  is_bound_(false)
{
  if (useCPUtime) {
    is_bound_ = CPUAffinity::bindToCurrentCpu();
    if (is_bound_)
      // the process is (now) bound to a single CPU, the call to clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...) is safe to use
      edm::LogInfo("TimerService") << "this process is bound to CPU " << CPUAffinity::currentCpu();
    else
      // the process is NOT bound to a single CPU
      edm::LogError("TimerService") << "this process is NOT bound to a single CPU, the results of the TimerService may be undefined";
  }

  iAR.watchPreModule(this, &TimerService::preModule);
  iAR.watchPostModule(this, &TimerService::postModule);
}

TimerService::~TimerService()
{
  if (useCPUtime and not is_bound_)
    std::cout << "this process is NOT bound to a single CPU, the results of the TimerService may be undefined";
  std::cout << "==========================================================\n";
  std::cout << " TimerService Info:\n";
  std::cout << " Used " << (useCPUtime ? "CPU" : "wall-clock") << "time for timing information\n";
  std::cout << "==========================================================\n";
  std::cout << std::flush;
}

// fwk calls this method before a module is processed
void TimerService::preModule(const edm::ModuleDescription& iMod)
{
  cpu_timer.reset();
  cpu_timer.start();  
}

// fwk calls this method after a module has been processed
void TimerService::postModule(const edm::ModuleDescription& iMod)
{
  cpu_timer.stop();
  double time = cpu_timer.delta();  // in secs
  newMeasurementSignal(iMod, time);
}
