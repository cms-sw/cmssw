#include "HLTrigger/Timer/interface/CPUTimerService.h"

edm::CPUTimer * CPUTimerService::cpu_timer = 0;

CPUTimerService::CPUTimerService(const edm::ParameterSet&, 
				 edm::ActivityRegistry& iAR)
{
  iAR.watchPreModule(this, &CPUTimerService::preModule);
  iAR.watchPostModule(this, &CPUTimerService::postModule);
  
  if(!cpu_timer)
    cpu_timer = new edm::CPUTimer();

  module_name = ""; cpu_time = 0;
}

CPUTimerService::~CPUTimerService()
{
  if(cpu_timer)
    {
      delete cpu_timer; cpu_timer = 0;
    }
}

// fwk calls this method before a module is processed
void CPUTimerService::preModule(const edm::ModuleDescription& iMod)
{
  module_name = iMod.moduleLabel();
  cpu_time = 0;
  cpu_timer->reset();
  cpu_timer->start();  
}

// fwk calls this method after a module has been processed
void CPUTimerService::postModule(const edm::ModuleDescription& iMod)
{
  cpu_timer->stop();
  cpu_time = cpu_timer->cpuTime();
  // do I need to check this?
  //  assert(iMod.moduleLabel() == module_name);
}
