#ifndef PERFSTAT_H
#define PERFSTAT_H

#include <linux/perf_event.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <unistd.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <x86intrin.h>
#include<cmath>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif




class PerfStat {
private:
  using Type=__u32;
  using Conf=__u64;

  static constexpr int METRIC_COUNT=7;
  static constexpr int METRIC_OFFSET=3;
  static constexpr int NGROUPS=4;


  Type types[NGROUPS][METRIC_COUNT] = {
    {
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      //    PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_RAW
      // PERF_TYPE_RAW
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_RAW,
      // PERF_TYPE_RAW,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    }
  };


  Conf confs[NGROUPS][METRIC_COUNT]= {
    {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_HW_INSTRUCTIONS,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      // 0xc488, // All indirect branches that are not calls nor returns.
      // 0xc888,   // All indirect return branches
      // 0xd088,  // All non-indirect calls executed.
      //    PERF_COUNT_HW_BUS_CYCLES,
      PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
      PERF_COUNT_HW_BRANCH_MISSES,
      0xc488 // All indirect branches that are not calls nor returns.
    }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_HW_INSTRUCTIONS,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      0x0114,   // ARITH.DIV_BUSY
      PERF_COUNT_HW_CACHE_REFERENCES,
      PERF_COUNT_HW_CACHE_MISSES
    }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_HW_INSTRUCTIONS,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      0x180010e,     // stall issued (front-end stalls)
      0x180ffa1,     // 0-ports (back-end stalls)
      0x18063a1,     // 0-exec-ports (back-end stalls)
      // 0x1a2,      //   res stall         
      // 0x0408,   // DTLB walk                                                       
      // 0x0485,  // ITLB walk
      // 0x02C2 // RETIRE_SLOTS
     }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_HW_INSTRUCTIONS,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      0x0280, // ICACHE.MISSES
      0x0151, // L1D.REPLACEMENT
      0x1a2     //   res stall         
      // 0x6000860     // off core outstanding > 6
    }
  };
  
  int fds[NGROUPS]={-1,};
  int cfds=-1; // current one
  int cgroup=NGROUPS-1;
  unsigned long long ncalls[NGROUPS]={0,};
  unsigned long long totcalls=0;
  long long times[2];
  // 0 seems needed  +1 is rdtsc +2 is gettime
  long long results[NGROUPS][METRIC_OFFSET+METRIC_COUNT+2];
 
  long long bias[NGROUPS][METRIC_OFFSET+METRIC_COUNT+2];
  long long bias1[METRIC_OFFSET+METRIC_COUNT+2];

  bool active=false;
  bool multiplex=false;
  
public:
  template<typename T> 
  static T sum(T const * t) { T s=0; for (int i=0; i!=NGROUPS; i++) s+=t[i]; return s;}
  
  static constexpr int ngroups() { return NGROUPS;}

  unsigned long long calls() const { return totcalls;}
  unsigned long long callsTot() const { return sum(ncalls);}
  
  double nomClock() const { return double(times[0])/double(times[1]); }
  double clock() const { return double(cyclesRaw())/double(taskTimeRaw()); }
  double turbo() const { return cyclesTot()/double(nomCyclesRaw());}

  // double corr(int i) const { return ( (0==ncalls[i]) | (0==results[i][2]) ) ? 0 : double(ncalls[i])*double(results[i][1])/(double(results[i][2])*double(callsTot()));}
  double corr(int i) const { return ( (0==ncalls[i]) | (0==results[i][2]) ) ? 0 : double(results[i][1])/(double(results[i][2])*(multiplex ? double(NGROUPS) : 1. ));}
  
  long long sum(int k) const { long long s=0; for (int i=0; i!=NGROUPS; i++) s+=results[i][k]; return s;}
  long long cyclesRaw() const { return sum(METRIC_OFFSET+0);}
  long long instructionsRaw() const { return sum(METRIC_OFFSET+1);}
  long long taskTimeRaw() const { return sum(METRIC_OFFSET+3);}
  long long realTimeRaw() const { return results[0][METRIC_OFFSET+METRIC_COUNT+1];}
  long long nomCyclesRaw() const { return results[0][METRIC_OFFSET+METRIC_COUNT+0];}
  
  
  double corrsum(int k) const { double s=0; for (int i=0; i!=NGROUPS; i++) s+=corr(i)*results[i][k]; return s;}
  double cyclesTot() const { return corrsum(METRIC_OFFSET+0);}
  double instructionsTot() const { return corrsum(METRIC_OFFSET+1);}
  double taskTimeTot() const { return corrsum(METRIC_OFFSET+3);}
  
  
  double cycles() const { return (0==calls()) ? 0 : cyclesTot()/double(calls()); }
  double instructions() const { return (0==calls()) ? 0 : instructionsTot()/double(calls()); }
  double taskTime() const { return (0==calls()) ? 0 : taskTimeTot()/double(calls()); }
  double realTime() const { return (0==calls()) ? 0 : double(results[0][METRIC_OFFSET+METRIC_COUNT+1])/double(calls()); }
  
  // instructions per cycle
  double ipc() const { return double(instructionsRaw())/double(cyclesRaw());}
  
  // fraction of branch instactions
  double brfrac() const { return double(results[0][METRIC_OFFSET+4])/double(results[0][METRIC_OFFSET+1]);}
  // missed branches per cycle
  double mbpc() const { return double(results[0][METRIC_OFFSET+5])/double(results[0][METRIC_OFFSET+0]);}
  
  double dtlbpc() const { return double(results[2][METRIC_OFFSET+4])/double(results[2][METRIC_OFFSET+0]);}
  double itlbpc() const { return double(results[2][METRIC_OFFSET+5])/double(results[2][METRIC_OFFSET+0]);}

  double stallFpc() const { return double(results[2][METRIC_OFFSET+4])/double(results[2][METRIC_OFFSET+0]);}
  double stallBpc() const { return double(results[2][METRIC_OFFSET+5])/double(results[2][METRIC_OFFSET+0]);}
  double stallEpc() const { return double(results[2][METRIC_OFFSET+6])/double(results[2][METRIC_OFFSET+0]);}

  // double rslotpc() const { return double(results[2][METRIC_OFFSET+6])/double(results[2][METRIC_OFFSET+0]);}
  
  
  // cache references per cycle
  double crpc() const { return double(results[1][METRIC_OFFSET+5])/double(results[1][METRIC_OFFSET+0]);}
  
  // main memory references (cache misses) per cycle
  double mrpc() const { return double(results[1][METRIC_OFFSET+6])/double(results[1][METRIC_OFFSET+0]);}
  
  // div-busy per cycle
  double  divpc() const { return double(results[1][METRIC_OFFSET+4])/double(results[1][METRIC_OFFSET+0]);}
  
  
  // L1 instruction-cache misses  (per cycles)
  double il1mpc() const { return double(results[3][METRIC_OFFSET+4])/double(results[0][METRIC_OFFSET+0]);}
  // L1 data-cache misses  (per cycles)
  double dl1mpc() const { return double(results[3][METRIC_OFFSET+5])/double(results[0][METRIC_OFFSET+0]);}
  // offcore full
  double offpc() const { return double(results[3][METRIC_OFFSET+6])/double(results[0][METRIC_OFFSET+0]);}



  // indirect calls  (per cycles)
  double icallpc() const { return double(results[0][METRIC_OFFSET+6])/double(results[0][METRIC_OFFSET+0]);}


  // fraction of bus cycles
  // double buspc() const { return double(results0[METRIC_OFFSET+4])/double(results0[METRIC_OFFSET+0]);}
  

  PerfStat(bool imultiplex=false) : multiplex(imultiplex){ init();}
  
  // share file descriptors...
  
  struct FD { int const * fds; bool mplex;};
  FD fd() const { FD f; f.fds=fds; f.mplex=multiplex; return f;}
  PerfStat(FD f) : multiplex(f.mplex) {
    totcalls=0;
    times[0]=times[1]=0;
    for (int k=0; k!=NGROUPS; k++) {
      for (int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) results[k][i]=bias[k][i]=0;
      ncalls[k]=0;
      fds[k] = f.fds[k];
    }
  }
  
  void init() {
    // pid_t id = getpid();
    pid_t id = 0;
    int cpuid=-1; int flags=0;	
    struct perf_event_attr pe;
    
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = types[0][0];
    pe.size = sizeof(struct perf_event_attr);
    pe.config = confs[0][0];
    pe.disabled = 1;
    pe.inherit=0;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    pe.mmap = 0;
    for (int k=0; k!=NGROUPS; k++) {
      pe.type = types[k][0];
      pe.config = confs[k][0];
      fds[k] = syscall(__NR_perf_event_open, &pe, id, cpuid, -1, flags);
    }
    pe.disabled = 0;
    
    // a small hack
    if (!isINTEL()) {
      confs[1][4] = PERF_COUNT_HW_BUS_CYCLES;
      types[1][4] = PERF_TYPE_HARDWARE;
    }
    
    // non exe uops
    confs[2][6] = isHaswell() ? 0x18063a1 : 0x18083a1;


    for (int k=0; k!=NGROUPS; k++) {
      for  (int i=1; i!=METRIC_COUNT; ++i) {
	pe.config = confs[k][i]; pe.type = types[k][i];
	int f = syscall(__NR_perf_event_open, &pe, id, cpuid, fds[k], flags);
	if (f==-1) std::cout << "error 1:" << i << " " << errno << " " << strerror(errno) << std::endl;
      }
      ioctl(fds[k], PERF_EVENT_IOC_RESET, 0);
    }
    
    totcalls=0;
    times[0]=times[1]=0;
    for (int k=0; k!=NGROUPS; k++) {
      for(int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) results[k][i]=bias[k][i]=0;
      ncalls[k]=0;
    }
    cgroup=NGROUPS-1;
    warmup();
  }
  
  ~PerfStat(){
    // don't! messes up the whole thing
    // ::close(fds0);
    // ::close(fds1);
  }
  
  void reset() {
    totcalls=0;
    for (int k=0; k!=NGROUPS; k++){
      ncalls[k]=0;
      for(int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) bias[0][i]+=results[0][i];
    }
    cgroup=NGROUPS-1;
    //    ::close(fds0);
    //::close(fds1);
    //init();
  }

  void start() {
    if(active) return;
    if (multiplex) return startAll();
    if((++cgroup)==NGROUPS) cgroup=0;
    start(cgroup);
  }
 
  void start(int k) {
    if(active) return;
    active=true;
    ++totcalls;
    ++ncalls[k];
    cfds =  fds[k];
    times[1] -= seconds();
    ioctl(cfds, PERF_EVENT_IOC_ENABLE, 0);
    times[0] -= rdtsc();
  }
 

 
  void startAll() {
    if(active) return;
    active=true;
    ++totcalls;
    for (int k=0; k!=NGROUPS; k++) {
      ++ncalls[k];
      ioctl(fds[k], PERF_EVENT_IOC_ENABLE, 0);
    }
    times[0] -= rdtsc();
  }
 
 
  void stop() {
    if (multiplex) return stopAll();
    times[0] += rdtsc();
    ioctl(cfds, PERF_EVENT_IOC_DISABLE, 0);
    times[1] += seconds();
    active=false;
    cfds=-1;
  }
 
  void stopAll() {
    times[0] += rdtsc();
    for (int k=0; k!=NGROUPS; k++)
      ioctl(fds[k], PERF_EVENT_IOC_DISABLE, 0);
    times[1] += seconds();
    active=false;
    cfds=-1;
  }
  

  int read() {
    long int ret=0;
    for (int k=0; k!=NGROUPS; k++)
      ret = std::min(ret,::read(fds[k], results[k], (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    results[0][METRIC_OFFSET+METRIC_COUNT]=times[0];results[0][METRIC_OFFSET+METRIC_COUNT+1]=times[1];
    return ret;
  }


  bool verify(double res) {
    read();calib();
    auto ok = [=](double x, double y) { return std::abs((x-y)/y)<res;};
    bool ret=true;
    for (int k=0; k!=NGROUPS; k++)
      if (ncalls[k]>0) 
	ret &= ok(results[k][1],results[k][METRIC_OFFSET+METRIC_COUNT+1]);

    ret &= ok(sum(2),results[0][METRIC_OFFSET+3]);
    return ret;
  }

  void warmup() {
    if(active) return;
    int nloop = multiplex ? 10 : 10*NGROUPS;
    for (int i=0; i!=nloop; ++i) {start();stop();}
    read();
    totcalls-=nloop; 
    for (int k=0; k!=NGROUPS; k++) {
      ncalls[k]-=10;
      for (int i=1; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) 
	bias[k][i] +=results[k][i];
    }
  }

  void calib() {
    if(active) return;
    
    int nloop = multiplex ? 10 : 10*NGROUPS;
    for (int i=0; i!=nloop; ++i) {start();stop();}
    totcalls-=nloop; 
    for (int k=0; k!=NGROUPS; k++) ncalls[k]-=10;
    
    long long results_c[NGROUPS][METRIC_COUNT+METRIC_OFFSET+2];
    long int err=0;
    for (int k=0; k!=NGROUPS; k++)
      err = std::min(err,::read(fds[k], results_c[k], (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    results_c[0][METRIC_OFFSET+METRIC_COUNT]=times[0];results_c[0][METRIC_OFFSET+METRIC_COUNT+1]=times[1];
    if (err==-1) return;
    for (int k=0; k!=NGROUPS; k++) {
      for (int i=1; i!=METRIC_OFFSET+METRIC_COUNT+2; ++i) {
	results_c[k][i]-=results[k][i];
	results[k][i] -= ncalls[k]*results_c[k][i]/10 + bias[k][i];
	// update bias for next read...
	bias[k][i] +=results_c[k][i];
      }
    }
  }
  

  void startDelta() {
    if(active) return;
    long long results_c[METRIC_COUNT+METRIC_OFFSET];
    long int err=0;
    for (int k=0; k!=NGROUPS; k++) {
      err = std::min(err,::read(fds[k], results_c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
      for (int i=0; i!=METRIC_OFFSET+METRIC_COUNT; ++i)
	results[k][i] -= results_c[i];
    }
    if (err==-1) return;
    start();
  }

  void stopDelta() {
    if(!active) return;
    stop();
    long long results_c[METRIC_COUNT+METRIC_OFFSET];
    long int err=0;
    for (int k=0; k!=NGROUPS; k++) {
      err = std::min(err,::read(fds[k], results_c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
      for (int i=0; i!=METRIC_OFFSET+METRIC_COUNT; ++i)
	results[k][i] += results_c[i];
    }
    results[0][METRIC_OFFSET+METRIC_COUNT]=times[0];results[0][METRIC_OFFSET+METRIC_COUNT+1]=times[1];
    if (err==-1) return;
  }



  static void header(std::ostream & out, bool details=false) {
    const char * sepF = "|  *"; 
    const char * sep = "*|  *"; 
    const char * sepL = "*|"; 
    out << sepF << "real time"
        << sep << "task time"
   	<< sep << "cycles" 
	<< sep << "ipc"
	<< sep << "br/ins"
	<< sep << "missed-br/cy"
	<< sep << "cache-ref/cy"
	<< sep << "mem-ref/cy"
	<< sep << "missed-L1I/cy"
	<< sep << "missed-L1D/cy"
	<< sep << "offcore/cy"
	<< sep <<  (isINTEL() ? "div/cy" : "bus/cy")
	<< sep << "ind-call/cy"
      // << sep << "dtlb-walk/cy"
      // << sep << "itlb-walk/cy"
     	<< sep << "front-stall/cy"
     	<< sep << "back-stall/cy"
     	<< sep << "exec-stall/cy"
      //	<< sep << "rslot/cy"
      //	<< sep << "bus/cy"
        << sep << "ncalls"
      ;
    if (details) {
      out << sep << "clock"
	  << sep << "turbo"
	  << sep << "multiplex";
    }
    out << sepL << std::endl;
  }
  
  void summary(std::ostream & out, bool details=false, double mult=1.e-6, double percent=100.) const {
    const char * sep = "|  "; 
    out << sep << mult*realTime() 
        << sep << mult*taskTime()
	<< sep << mult*cycles() 
	<< sep << ipc()
	<< sep << percent*brfrac()
	<< sep << percent*mbpc()
	<< sep << percent*crpc()
	<< sep << percent*mrpc()
      	<< sep << percent*il1mpc()
      	<< sep << percent*dl1mpc()
      	<< sep << percent*offpc()
	<< sep << percent*divpc()
	<< sep << percent*icallpc()
      // 	<< sep << percent*1000.*1000.*dtlbpc()
      // 	<< sep << percent*1000.*1000.*itlbpc()
	<< sep << percent*stallFpc()
	<< sep << percent*stallBpc()
	<< sep << percent*stallEpc()
      // 	<< sep << rslotpc()
      // buspc()
      << sep << calls()
      ;
    if (details) {
      out << sep << clock()
	  << sep << turbo() << sep;
      for (int k=0; k!=NGROUPS; k++) { out << ncalls[k] <<'/'<<percent/corr(k) <<",";}
    }
    out << sep << std::endl;
  }

  void print(std::ostream & out, bool docalib=true, bool debug=false) {
    if (-1==read()) out << "error in reading" << std::endl;
    if (docalib) calib();  

    summary(out,true);
    
    if (!debug) return;

    out << double(results[0][METRIC_OFFSET+METRIC_COUNT])/double(results[0][METRIC_OFFSET+METRIC_COUNT+1]) 
	<< " "<<  double(results[0][METRIC_OFFSET+0])/double(results[1][METRIC_OFFSET+1]) 
	<< " "<<  double(cyclesRaw())/double(results[0][METRIC_OFFSET+METRIC_COUNT])
	<< " "<<  double(taskTimeRaw())/double(results[0][METRIC_OFFSET+METRIC_COUNT+1]) << std::endl;
    for (int k=0; k!=NGROUPS; k++) {
      out << ncalls[k] << " ";
      for (int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i)  out << results[k][i] << " ";
      out << "; " <<  double(results[k][METRIC_OFFSET+0])/double(results[k][METRIC_OFFSET+3]) 
	  << " " << double(results[k][METRIC_OFFSET+1])/double(results[k][METRIC_OFFSET+0])
	  << " " << double(results[k][2])/double(results[k][1]);
      out  << std::endl;
    }
    out  << std::endl;
    
  }

  static long long seconds() {
    struct timespec ts;
    
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
  
    return (long long)(ts.tv_sec)*1000000000LL + ts.tv_nsec;
}

  static volatile unsigned long long rdtsc() {
    unsigned int taux=0;
    return __rdtscp(&taux);
  }

  static bool isHaswell() {
    return modelNumber()==0x3c; 
  }
  
  static unsigned int modelNumber() {
    unsigned int eax;
    cpuid(1, &eax, nullptr, nullptr, nullptr);
    // return eax;
   return ( (eax&0xf0) >> 4) + ( (eax&0xf0000) >> 12);
  }

  static bool isINTEL() {
    char v[13] = { 0, };
    unsigned int cpuid_level=0;
    cpuid(0, &cpuid_level, (unsigned int *)&v[0], ( unsigned int *)&v[8], ( unsigned int *)&v[4]);

    return 0==::strcmp(v,"GenuineIntel");
  }
  
  static void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx, unsigned int *ecx, unsigned int *edx) {
    unsigned int a = eax ? *eax : 0;
    unsigned int b = ebx ? *ebx : 0;
    unsigned int c = ecx ? *ecx : 0;
    unsigned int d = edx ? *edx : 0;
    
#if defined __i386__
    __asm__ __volatile__ ("xchgl	%%ebx,%0\n\t"
			  "cpuid	\n\t"
			  "xchgl	%%ebx,%0\n\t"
			  : "+r" (b), "=a" (a), "=c" (c), "=d" (d)
			  : "1" (op), "2" (c));
#else
    __asm__ __volatile__ ("cpuid"
			  : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
			  : "0" (op), "2" (c));
#endif
    
    if (eax) *eax = a;
    if (ebx) *ebx = b;
    if (ecx) *ecx = c;
    if (edx) *edx = d;
  }

};


#endif
