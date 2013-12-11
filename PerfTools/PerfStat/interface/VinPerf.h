#ifndef VinPerf_H
#define VinPerf_H

#include "PerfStatBase.h"


// VI preferred...
class VinPerf final : public PerfStatBase<4>{

 private:


  Type types[NGROUPS][METRIC_COUNT] = {
    {
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_HARDWARE,
      //    PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_RAW
      // PERF_TYPE_RAW
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_RAW,
      // PERF_TYPE_RAW,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    }, { 
      PERF_TYPE_HARDWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_SOFTWARE,
      PERF_TYPE_HARDWARE,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    }
  };


  Conf confs[NGROUPS][METRIC_COUNT]= {
    {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      PERF_COUNT_HW_INSTRUCTIONS,
      // 0xc488, // All indirect branches that are not calls nor returns.
      // 0xc888,   // All indirect return branches
      // 0xd088,  // All non-indirect calls executed.
      //    PERF_COUNT_HW_BUS_CYCLES,
      PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
      PERF_COUNT_HW_BRANCH_MISSES,
      0xc488 // All indirect branches that are not calls nor returns.
    }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      PERF_COUNT_HW_INSTRUCTIONS,
      0x0114,   // ARITH.DIV_BUSY
      PERF_COUNT_HW_CACHE_REFERENCES,
      PERF_COUNT_HW_CACHE_MISSES
    }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      PERF_COUNT_HW_INSTRUCTIONS,
      0x180010e,     // stall issued (front-end stalls)
      0x180ffa1,     // 0-ports (back-end stalls)
      0x18063a1,     // 0-exec-ports (back-end stalls)
      // 0x1a2,      //   res stall         
      // 0x0408,   // DTLB walk                                                       
      // 0x0485,  // ITLB walk
      // 0x02C2 // RETIRE_SLOTS
     }, {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,
      PERF_COUNT_HW_INSTRUCTIONS,
      0x0280, // ICACHE.MISSES
      0x0151, // L1D.REPLACEMENT
      0x1a2     //   res stall         
      // 0x6000860     // off core outstanding > 6
    }
  };

public:

  VinPerf (bool imultiplex=false) : PerfStatBase<4>(imultiplex){init();}
  VinPerf(PerfStatBase::FD f) : PerfStatBase<4>(f){}

  void get(Conf * c, Type * t) const {
    memcpy(c,&confs[0][0], NGROUPS*METRIC_COUNT*sizeof(Conf));
    memcpy(t,&types[0][0], NGROUPS*METRIC_COUNT*sizeof(Type));

  };

  long long instructionsRaw() const { return sum(METRIC_OFFSET+3);}
  double instructionsTot() const { return corrsum(METRIC_OFFSET+3);}

  double instructions() const { return (0==calls()) ? 0 : instructionsTot()/double(calls()); }
   // instructions per cycle
  double ipc() const { return double(instructionsRaw())/double(cyclesRaw());}
  
  // fraction of branch instactions
  double brfrac() const { return double(results[0][METRIC_OFFSET+4])/double(results[0][METRIC_OFFSET+3]);}
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

  void header(std::ostream & out, bool details=false) const {
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

};


#endif
