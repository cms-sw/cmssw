#include <iostream>
#include <sched.h>

#if defined(__linux__) && ! __GLIBC_PREREQ(2, 6)

// CPU_COUNT is not defined in glibc 2.5, use a gcc builtin instead

size_t cpu_count(cpu_set_t const * cpu_set) {
  // cpu_set_t is an array of__cpu_mask, a typedef for unsigned long int
  size_t count = 0;
  for (unsigned int i = 0; i < sizeof(cpu_set_t) / sizeof(__cpu_mask); ++i)
    count += __builtin_popcountl(cpu_set->__bits[i]);
  return count;
}
#define CPU_COUNT(CPU_SET) cpu_count(CPU_SET)

// sched_getcpu is not defined in glibc 2.5, use a syscall to work around it
// code adapted from IcedTea

#if defined(__x86_64__)
#include <asm/vsyscall.h>
#elif defined(__i386__)                                                                                                                                                                                                          
#include <sys/syscall.h>
#endif

static int sched_getcpu(void) {
  unsigned int cpu;
  int retval = -1;
#if defined(__x86_64__)
  typedef long (*vgetcpu_t)(unsigned int *cpu, unsigned int *node, unsigned long *tcache);
  vgetcpu_t vgetcpu = (vgetcpu_t)VSYSCALL_ADDR(__NR_vgetcpu);
  retval = vgetcpu(&cpu, NULL, NULL);
#elif defined(__i386__)
  retval = syscall(SYS_getcpu, &cpu, NULL, NULL);
#endif

  return (retval == -1) ? retval : cpu;
}

#endif // __linux__ && ! __GLIBC_PREREQ(2, 6))

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/Timer/interface/TimerService.h"

TimerService::TimerService(const edm::ParameterSet& ps, 
                                 edm::ActivityRegistry& iAR) :
  useCPUtime( ps.getUntrackedParameter<bool>("useCPUtime", true) ),
  cpu_timer(useCPUtime),
  is_bound_(false)
{
#if defined __linux__
  // check if this process is bound to a single CPU, and try to bind if it's not
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set_t), & cpu_set);
  if (CPU_COUNT(& cpu_set) != 1) {
    // this process is unbound, bind it to the current CPU
    int current = sched_getcpu();
    CPU_ZERO(& cpu_set);
    CPU_SET(current, & cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), & cpu_set); // check for errors ?
    sched_getaffinity(0, sizeof(cpu_set_t), & cpu_set);
  }
  if (CPU_COUNT(& cpu_set) == 1) {
    // the process is (now) bound to a single CPU, the clock_gettime calls are safe to use
    is_bound_ = true;
    edm::LogInfo("TimerService") << "this process is bound to CPU " << sched_getcpu();
  } else {
    // the process is NOT bound to a single CPU
    is_bound_ = false;
    edm::LogError("TimerService") << "this process is NOT bound to a single CPU, the results of the TimerService may be undefined";
  }
#else
  // cpu affinity is currently only supposted on LINUX
  is_bound_ = false;
  edm::LogError("TimerService") << "this process is NOT bound to a single CPU, the results of the TimerService may be undefined";
#endif

  iAR.watchPreModule(this, &TimerService::preModule);
  iAR.watchPostModule(this, &TimerService::postModule);
}

TimerService::~TimerService()
{
  if (not is_bound_)
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
