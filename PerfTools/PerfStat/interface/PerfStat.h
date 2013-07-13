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

  Type types1[METRIC_COUNT] = { 
    PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE,
    PERF_TYPE_SOFTWARE,
    PERF_TYPE_SOFTWARE,
    //    PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE,
    PERF_TYPE_RAW
    // PERF_TYPE_RAW
  };

  Type types2[METRIC_COUNT] = { 
    PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE,
    PERF_TYPE_SOFTWARE,
    PERF_TYPE_SOFTWARE,
    PERF_TYPE_RAW,
    // PERF_TYPE_RAW,
    PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE
  };


  Conf confs1[METRIC_COUNT]= {
    PERF_COUNT_HW_CPU_CYCLES,
    PERF_COUNT_HW_INSTRUCTIONS,
    PERF_COUNT_SW_CPU_CLOCK,
    PERF_COUNT_SW_TASK_CLOCK,
    // 0x0488, // BR_INST_EXEC:INDIRECT_NON_CALL
    //    PERF_COUNT_HW_BUS_CYCLES,
    PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
    PERF_COUNT_HW_BRANCH_MISSES,
    // 0x0408,   // DTLB walk                                                       
    //  0x0485  // ITLB walk
    0x0280 // ICACHE.MISSES
    // 0x02C2 // RETIRE_SLOTS
  };

  Conf confs2[METRIC_COUNT]= {
    PERF_COUNT_HW_CPU_CYCLES,
    PERF_COUNT_HW_INSTRUCTIONS,
    PERF_COUNT_SW_CPU_CLOCK,
    PERF_COUNT_SW_TASK_CLOCK,
    0x0114,   // ARITH.DIV_BUSY
    PERF_COUNT_HW_CACHE_REFERENCES,
    PERF_COUNT_HW_CACHE_MISSES
  };

  int fds0=0;
  int fds1=0;
  int fds=0;
  unsigned long long ncalls=0;
  unsigned long long ncalls0=0;
  unsigned long long ncalls1=0;
  long long times[2];
  // 0 seems needed  +1 is rdtsc +2 is gettime
  long long results0[METRIC_OFFSET+METRIC_COUNT+2];
  long long results1[METRIC_OFFSET+METRIC_COUNT+2];

  long long bias0[METRIC_OFFSET+METRIC_COUNT+2];
  long long bias1[METRIC_OFFSET+METRIC_COUNT+2];

  bool active=false;
  
public:
  unsigned long long calls() const { return ncalls;}
  unsigned long long callsTot() const { return ncalls0+ncalls1;}

  double nomClock() const { return double(times[0])/double(times[1]); }
  double clock() const { return double(cyclesRaw())/double(taskTimeRaw()); }

  double corr0() const { return ( (0==ncalls0) | (0==results0[2]) ) ? 0 : double(calls())*double(results0[1])/(double(results0[2])*double(callsTot()));}
  double corr1() const { return ( (0==ncalls1) | (0==results1[2]) ) ? 0 : double(calls())*double(results1[1])/(double(results1[2])*double(callsTot()));}

  long long cyclesRaw() const { return results0[METRIC_OFFSET+0]+results1[METRIC_OFFSET+0];}
  long long instructionsRaw() const { return results0[METRIC_OFFSET+1]+results1[METRIC_OFFSET+1];}
  long long taskTimeRaw() const { return results0[METRIC_OFFSET+3]+results1[METRIC_OFFSET+3];}
  long long realTimeRaw() const { return results0[METRIC_OFFSET+METRIC_COUNT+1];}


  double cyclesTot() const { return corr0()*results0[METRIC_OFFSET+0]+corr1()*results1[METRIC_OFFSET+0];}
  double instructionsTot() const { return corr0()*results0[METRIC_OFFSET+1]+corr1()*results1[METRIC_OFFSET+1];}
  double taskTimeTot() const { return corr0()*results0[METRIC_OFFSET+3]+corr1()*results1[METRIC_OFFSET+3];}


  double cycles() const { return (0==ncalls) ? 0 : cyclesTot()/double(ncalls); }
  double instructions() const { return (0==ncalls) ? 0 : instructionsTot()/double(ncalls); }
  double taskTime() const { return (0==ncalls) ? 0 : taskTimeTot()/double(ncalls); }
  double realTime() const { return (0==ncalls) ? 0 : double(results0[METRIC_OFFSET+METRIC_COUNT+1])/double(ncalls); }

  // instructions per cycle
  double ipc() const { return double(instructionsRaw())/double(cyclesRaw());}
  
  // fraction of branch instactions
  double brfrac() const { return double(results0[METRIC_OFFSET+4])/double(results0[METRIC_OFFSET+1]);}
  // missed branches per cycle
  double mbpc() const { return double(results0[METRIC_OFFSET+5])/double(results0[METRIC_OFFSET+0]);}

  // double dtlbpc() const { return double(results0[METRIC_OFFSET+6])/double(results0[METRIC_OFFSET+0]);}
  // double itlbpc() const { return double(results0[METRIC_OFFSET+7])/double(results0[METRIC_OFFSET+0]);}


  // cache references per cycle
  double crpc() const { return double(results1[METRIC_OFFSET+5])/double(results1[METRIC_OFFSET+0]);}

  // main memory references (cache misses) per cycle
  double mrpc() const { return double(results1[METRIC_OFFSET+6])/double(results1[METRIC_OFFSET+0]);}

  // div-busy per cycle
  double  divpc() const { return double(results1[METRIC_OFFSET+4])/double(results1[METRIC_OFFSET+0]);}


  // L1 instruction-cache misses  (per cycles)
  double il1mpc() const { return double(results0[METRIC_OFFSET+6])/double(results0[METRIC_OFFSET+0]);}

  // fraction of bus cycles
  // double buspc() const { return double(results0[METRIC_OFFSET+4])/double(results0[METRIC_OFFSET+0]);}


  PerfStat() { init();}

  // share file descriptors...

  struct FD { int f0; int f1;};
  FD fd() const { FD f{fds0,fds1}; return f;}
  PerfStat(FD f) : fds0(f.f0), fds1(f.f1) {
    ncalls=ncalls0=ncalls1=0;
    times[0]=times[1]=0;
    for	(int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) results0[i]=results1[i]=bias0[i]=bias1[i]=0;
  }
    

  void init() {
    pid_t id = getpid();
    int cpuid=-1; int flags=0;	
    struct perf_event_attr pe;
    
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = types1[0];
    pe.size = sizeof(struct perf_event_attr);
    pe.config = confs1[0];
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    pe.mmap = 0;
    
    fds0 = syscall(__NR_perf_event_open, &pe, id, cpuid, -1, flags);
    pe.type = types2[0];
    pe.config = confs2[0];
    fds1 = syscall(__NR_perf_event_open, &pe, id, cpuid, -1, flags);
    pe.disabled = 0;
    
    for	(int i=1; i!=METRIC_COUNT; ++i) {
      pe.config = confs1[i]; pe.type = types1[i];
      int fds = syscall(__NR_perf_event_open, &pe, id, cpuid, fds0, flags);
      if (fds==-1) std::cout << "error 0:" << i << " " << errno << " " << strerror(errno) << std::endl;
    }

    // a small hack
    if (!isINTEL()) {
      confs2[4] = PERF_COUNT_HW_BUS_CYCLES;
      types2[4] = PERF_TYPE_HARDWARE;
    }
    
    for  (int i=1; i!=METRIC_COUNT; ++i) {
      pe.config = confs2[i]; pe.type = types2[i];
      int fds = syscall(__NR_perf_event_open, &pe, id, cpuid, fds1, flags);
      if (fds==-1) std::cout << "error 1:" << i << " " << errno << " " << strerror(errno) << std::endl;
    }
    
    ioctl(fds0, PERF_EVENT_IOC_RESET, 0);
    ioctl(fds1, PERF_EVENT_IOC_RESET, 0);
    ncalls=ncalls0=ncalls1=0;
    times[0]=times[1]=0;
    for	(int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) results0[i]=results1[i]=bias0[i]=bias1[i]=0;

    warmup();
  }

  ~PerfStat(){
    // don't! messes up the whole thing
    // ::close(fds0);
    // ::close(fds1);
  }
  
  void reset() {
    ncalls=ncalls0=ncalls1=0;
    for	(int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) {
      bias0[i]+=results0[i];
      bias1[i]+=results1[i];
    }
    //    ::close(fds0);
    //::close(fds1);
    //init();
  }

  void start() {
    if(active) return;
    active=true;
    ++ncalls;
    if ((ncalls&1)==0) ++ncalls0; else ++ncalls1;
    fds = ( (ncalls&1)==0) ? fds0 : fds1;
    times[1] -= seconds();
    ioctl(fds, PERF_EVENT_IOC_ENABLE, 0);
    times[0] -= rdtsc();
  }
 
  void start0() {
    if(active) return;
    active=true;
    ++ncalls;
    ++ncalls0;
    fds =  fds0;
    times[1] -= seconds();
    ioctl(fds, PERF_EVENT_IOC_ENABLE, 0);
    times[0] -= rdtsc();
  }
 
  void start1() {
    if(active) return;
    active=true;
    ++ncalls;
    ++ncalls1;
    fds =  fds1;
    times[1] -= seconds();
    ioctl(fds, PERF_EVENT_IOC_ENABLE, 0);
    times[0] -= rdtsc();
  }
 
  void startAll() {
    if(active) return;
    active=true;
    ++ncalls;
    ++ncalls0;
    ++ncalls1;
    times[1] -= seconds();
    ioctl(fds0, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fds1, PERF_EVENT_IOC_ENABLE, 0);
    times[0] -= rdtsc();
  }
 
 
  void stop() {
    times[0] += rdtsc();
    ioctl(fds, PERF_EVENT_IOC_DISABLE, 0);
    times[1] += seconds();
    active=false;
    fds=-1;
  }
 
  void stopAll() {
    times[0] += rdtsc();
    ioctl(fds1, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fds0, PERF_EVENT_IOC_DISABLE, 0);
    times[1] += seconds();
    active=false;
    fds=-1;
  }
  

  int read() {
    auto ret = ::read(fds0, results0, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long));
    results0[METRIC_OFFSET+METRIC_COUNT]=times[0];results0[METRIC_OFFSET+METRIC_COUNT+1]=times[1];
    ret = std::min(ret,::read(fds1, results1, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    return ret;
  }


  bool verify(double res) {
    read();calib();
    auto ok = [=](double x, double y) { return std::abs((x-y)/y)<res;};
    bool ret=true;
    if (ncalls0>0) {
      ret &= ok(results0[1],results0[METRIC_OFFSET+METRIC_COUNT+1]);
    }
    if (ncalls1>1) {
      ret &= ok(results1[1],results1[METRIC_OFFSET+METRIC_COUNT+1]);
    }
    ret &= ok(results0[2]+results1[2],results0[METRIC_OFFSET+3]);
    return ret;
  }

  void warmup() {
   if(active) return;
    for (int i=0; i!=20; ++i) {start();stop();}
    read();
    ncalls-=20; ncalls0-=10;ncalls1-=10;
    for (int i=1; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i) {
      bias0[i] +=results0[i];
      bias1[i] +=results1[i];
    }
  }

  void calib() {
    if(active) return;

    for (int i=0; i!=20; ++i) {start();stop();}
    ncalls-=20; ncalls0-=10;ncalls1-=10;

    long long results0c[METRIC_COUNT+METRIC_OFFSET+2];
    long long results1c[METRIC_COUNT+METRIC_OFFSET+2];
    auto err = ::read(fds0, results0c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long));
    err = std::min(err,::read(fds1, results1c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    results0c[METRIC_OFFSET+METRIC_COUNT]=times[0];results0c[METRIC_OFFSET+METRIC_COUNT+1]=times[1];
    if (err==-1) return;
    for (int i=1; i!=METRIC_OFFSET+METRIC_COUNT+2; ++i) {
      results0c[i]-=results0[i];
      results1c[i]-=results1[i];
      results0[i] -= ncalls0*results0c[i]/10 + bias0[i];
      results1[i] -= ncalls1*results1c[i]/10 + bias1[i];
      // update bias for next read...
      bias0[i] +=results0c[i];
      bias1[i] +=results1c[i];
    }
  }
  

  void startDelta() {
   if(active) return;
    long long results0c[METRIC_COUNT+METRIC_OFFSET];
    long long results1c[METRIC_COUNT+METRIC_OFFSET];
    auto err = ::read(fds0, results0c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long));
    err = std::min(err,::read(fds1, results1c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    if (err==-1) return;
    for (int i=0; i!=METRIC_OFFSET+METRIC_COUNT; ++i) {
      results0[i] -= results0c[i];
      results1[i] -= results1c[i];
    }
    start();
  }

  void stopDelta() {
   if(!active) return;
    stop();
    long long results0c[METRIC_COUNT+METRIC_OFFSET];
    long long results1c[METRIC_COUNT+METRIC_OFFSET];
    auto err = ::read(fds0, results0c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long));
    err = std::min(err,::read(fds1, results1c, (METRIC_OFFSET+METRIC_COUNT)*sizeof(long long)));
    if (err==-1) return;
    for (int i=0; i!=METRIC_OFFSET+METRIC_COUNT; ++i) {
      results0[i] += results0c[i];
      results1[i] += results1c[i];
    }
  }



  static void header(std::ostream & out) {
    const char * sepF = "|  *"; 
    const char * sep = "*|  *"; 
    const char * sepL = "*|"; 
    out << sepF << "clock"
	<< sep << "time"    
	<< sep << "cycles" 
	<< sep << "ipc"
	<< sep << "br/ins"
	<< sep << "missed-br/cy"
	<< sep << "cache-ref/cy"
	<< sep << "mem-ref/cy"
	<< sep <<  (isINTEL() ? "div/cy" : "bus/cy")
	<< sep << "missed-L1I/cy"
      //	<< sep << "dtlb-walk/cy"
      //	<< sep << "itlb-walk/cy"
      //	<< sep << "bus/cy"
	<< sepL << std::endl;
  }
  
  void summary(std::ostream & out, double mult=1.e-6, double percent=100.) const {
    const char * sep = "|  "; 
    out << sep << clock()  
	<< sep << mult*taskTime() 
	<< sep << mult*cycles() 
	<< sep << ipc()
	<< sep << percent*brfrac()
	<< sep << percent*mbpc()
	<< sep << percent*crpc()
	<< sep << percent*mrpc()
	<< sep << percent*divpc()
	<< sep << percent*il1mpc()
      // 	<< sep << dtlbpc()
      // 	<< sep << itlbpc()
      // buspc()
	<< sep << std::endl;
  }

  void print(std::ostream & out, bool docalib=true, bool debug=false) {
    if (-1==read()) out << "error in reading" << std::endl;
    if (docalib) calib();  

    summary(out);
    
    if (!debug) return;

    out << double(results0[METRIC_OFFSET+METRIC_COUNT])/double(results0[METRIC_OFFSET+METRIC_COUNT+1]) 
	<< " "<<  double(results0[METRIC_OFFSET+0])/double(results1[METRIC_OFFSET+1]) 
	<< " "<<  double(cyclesRaw())/double(results0[METRIC_OFFSET+METRIC_COUNT])
	<< " "<<  double(taskTimeRaw())/double(results0[METRIC_OFFSET+METRIC_COUNT+1]) << std::endl;
    out << ncalls0 << " ";
    for (int i=0; i!=METRIC_COUNT+METRIC_OFFSET+2; ++i)  out << results0[i] << " ";
    out << "; " <<  double(results0[METRIC_OFFSET+0])/double(results0[METRIC_OFFSET+3]) 
	<< " " << double(results0[METRIC_OFFSET+1])/double(results0[METRIC_OFFSET+0])
	<< " " << double(results0[2])/double(results0[1]);
    out  << std::endl;
    out << ncalls1 << " ";
    for (int i=0; i!=METRIC_COUNT+METRIC_OFFSET; ++i)  out << results1[i] << " ";
    out  << "; "<< double(results1[METRIC_OFFSET+0])/double(results1[METRIC_OFFSET+3]) 
	<< " " << double(results1[METRIC_OFFSET+1])/double(results1[METRIC_OFFSET+0])
	<< " " << double(results1[2])/double(results1[1]);

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
