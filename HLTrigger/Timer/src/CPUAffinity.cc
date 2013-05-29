#include "HLTrigger/Timer/interface/CPUAffinity.h"

#include <cstdlib>

#ifdef __linux

#include <sched.h>

#if ! __GLIBC_PREREQ(2, 6)
// CPU_COUNT is not defined in glibc 2.5, use a gcc builtin instead

static
size_t cpu_count(cpu_set_t const * cpu_set) 
{
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

static 
int sched_getcpu(void) 
{
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

#endif // ! __GLIBC_PREREQ(2, 6)
#endif // __lixnux


int CPUAffinity::currentCpu() {
  // cpu affinity is currently only supposted on LINUX
#ifdef __linux
  return sched_getcpu();
#else
  return 0;
#endif
}

bool CPUAffinity::isCpuBound()
{
  // cpu affinity is currently only supposted on LINUX
#ifdef __linux

  // check if this process is bound to a single CPU
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set_t), & cpu_set);
  if (CPU_COUNT(& cpu_set) == 1)
    // the process is bound to a single CPU
    return true;

#endif // __linux

  return false;
}


bool CPUAffinity::bindToCurrentCpu()
{
  // cpu affinity is currently only supposted on LINUX
#ifdef __linux

  // check if this process is bound to a single CPU, and try to bind to the current one if it's not
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set_t), & cpu_set);
  if (CPU_COUNT(& cpu_set) == 1)
    // the process is already bound to a single CPU
    return true;
  
  // this process is not bound, try to bind it to the current CPU
  int current = sched_getcpu();
  CPU_ZERO(& cpu_set);
  CPU_SET(current, & cpu_set);
  sched_setaffinity(0, sizeof(cpu_set_t), & cpu_set); // check for errors ?
  sched_getaffinity(0, sizeof(cpu_set_t), & cpu_set);

  if (CPU_COUNT(& cpu_set) == 1)
    // the process is now bound to a single CPU
    return true;

#endif // __linux

  return false;
}
