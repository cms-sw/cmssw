#include "HLTrigger/Timer/interface/TimerService.h"
#include <iostream>

TimerService::TimerService(const edm::ParameterSet& ps, 
				 edm::ActivityRegistry& iAR) :
  useCPUtime( ps.getUntrackedParameter<bool>("useCPUtime", true) ),
  cpu_timer( useCPUtime )
{
  iAR.watchPreModule(this, &TimerService::preModule);
  iAR.watchPostModule(this, &TimerService::postModule);
}

TimerService::~TimerService()
{
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
