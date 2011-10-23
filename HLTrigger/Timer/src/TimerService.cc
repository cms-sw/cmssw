#include "HLTrigger/Timer/interface/TimerService.h"
#include <iostream>

edm::CPUTimer * TimerService::cpu_timer = 0;

TimerService::TimerService(const edm::ParameterSet& ps, 
				 edm::ActivityRegistry& iAR)
{
  // whether to use CPU-time (default) or wall-clock time
  useCPUtime = ps.getUntrackedParameter<bool>("useCPUtime", true);

  iAR.watchPreModule(this, &TimerService::preModule);
  iAR.watchPostModule(this, &TimerService::postModule);
  
  if(!cpu_timer)
    cpu_timer = new edm::CPUTimer();

}

TimerService::~TimerService()
{
  if(cpu_timer){
    std::cout << "==========================================================\n";
    std::cout << " TimerService Info:\n";
    std::cout << " Used " << (useCPUtime ? "CPU" : "wall-clock") << "time for timing information\n";
    std::cout << "==========================================================\n";
    std::cout << std::flush;
    
    delete cpu_timer; cpu_timer = 0;
  }
}

// fwk calls this method before a module is processed
void TimerService::preModule(const edm::ModuleDescription& iMod)
{
  cpu_timer->reset();
  cpu_timer->start();  
}

// fwk calls this method after a module has been processed
void TimerService::postModule(const edm::ModuleDescription& iMod)
{
  cpu_timer->stop();

  double time = -999; // in secs
  if(useCPUtime)
    time = cpu_timer->cpuTime();
  else
    time = cpu_timer->realTime();

  newMeasurementSignal(iMod, time);
}
