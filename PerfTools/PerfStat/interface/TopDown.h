#ifndef TopDown_H
#define TopDonw_H

#include "PerfStatBase.h"


// Performance Monitoring Events for 3rd Generation Intel Core Processors Code Name IvyTown-IVT V7 8/16/2013 1:32:19 PM
class TopDown  final : public PerfStatBase<4> {
public:


  static constexpr unsigned int CODE_ARITH__FPU_DIV_ACTIVE = 0x530114;
  static constexpr unsigned int CODE_BACLEARS__ANY = 0x531FE6;
  static constexpr unsigned int CODE_BR_INST_RETIRED__NEAR_TAKEN = 0x5320C4;
  static constexpr unsigned int CODE_BR_MISP_RETIRED__ALL_BRANCHES = 0x5300C5;
  static constexpr unsigned int CODE_CPU_CLK_UNHALTED__REF_TSC = 0x530300;
  static constexpr unsigned int CODE_CPU_CLK_UNHALTED__THREAD = 0x530200;
  static constexpr unsigned int CODE_CYCLE_ACTIVITY__CYCLES_NO_EXECUTE = 0x45304A3;
  static constexpr unsigned int CODE_CYCLE_ACTIVITY__STALLS_L1D_PENDING = 0xC530CA3;
  static constexpr unsigned int CODE_CYCLE_ACTIVITY__STALLS_L2_PENDING = 0x55305A3;
  static constexpr unsigned int CODE_CYCLE_ACTIVITY__STALLS_LDM_PENDING = 0x65306A3;
  static constexpr unsigned int CODE_DSB2MITE_SWITCHES__PENALTY_CYCLES = 0x5302AB;
  static constexpr unsigned int CODE_FP_COMP_OPS_EXE__SSE_PACKED_DOUBLE = 0x531010;
  static constexpr unsigned int CODE_FP_COMP_OPS_EXE__SSE_PACKED_SINGLE = 0x534010;
  static constexpr unsigned int CODE_FP_COMP_OPS_EXE__SSE_SCALAR_DOUBLE = 0x538010;
  static constexpr unsigned int CODE_FP_COMP_OPS_EXE__SSE_SCALAR_SINGLE = 0x532010;
  static constexpr unsigned int CODE_FP_COMP_OPS_EXE__X87 = 0x530110;
  static constexpr unsigned int CODE_ICACHE__IFETCH_STALL = 0x530480;
  static constexpr unsigned int CODE_IDQ__ALL_DSB_CYCLES_4_UOPS = 0x4531879;
  static constexpr unsigned int CODE_IDQ__ALL_DSB_CYCLES_ANY_UOPS = 0x1531879;
  static constexpr unsigned int CODE_IDQ__ALL_MITE_CYCLES_4_UOPS = 0x4532479;
  static constexpr unsigned int CODE_IDQ__ALL_MITE_CYCLES_ANY_UOPS = 0x1532479;
  static constexpr unsigned int CODE_IDQ__MS_SWITCHES = 0x1573079;
  static constexpr unsigned int CODE_IDQ__MS_UOPS = 0x533079;
  static constexpr unsigned int CODE_IDQ_UOPS_NOT_DELIVERED__CORE = 0x53019C;
  static constexpr unsigned int CODE_IDQ_UOPS_NOT_DELIVERED__CYCLES_0_UOPS_DELIV__CORE = 0x453019C;
  static constexpr unsigned int CODE_ILD_STALL__LCP = 0x530187;
  static constexpr unsigned int CODE_INST_RETIRED__ANY = 0x5300C0;//  0x530100;
  static constexpr unsigned int CODE_INT_MISC__RECOVERY_CYCLES = 0x153030D;
  static constexpr unsigned int CODE_ITLB_MISSES__WALK_DURATION = 0x530485;
  static constexpr unsigned int CODE_L1D_PEND_MISS__PENDING = 0x530148;
  static constexpr unsigned int CODE_L1D_PEND_MISS__PENDING_CYCLES = 0x1530148;
  static constexpr unsigned int CODE_LSD__CYCLES_4_UOPS = 0x45301A8;
  static constexpr unsigned int CODE_LSD__CYCLES_ACTIVE = 0x15301A8;
  static constexpr unsigned int CODE_LSD__UOPS = 0x5301A8;
  static constexpr unsigned int CODE_MACHINE_CLEARS__COUNT  = 0x15701C3;
  static constexpr unsigned int CODE_MEM_LOAD_UOPS_RETIRED__L1_MISS = 0x5308D1;
  static constexpr unsigned int CODE_MEM_LOAD_UOPS_RETIRED__LLC_HIT = 0x5304D1;
  static constexpr unsigned int CODE_MEM_LOAD_UOPS_RETIRED__LLC_MISS = 0x5320D1;
  static constexpr unsigned int CODE_RESOURCE_STALLS__SB = 0x5308A2;
  static constexpr unsigned int CODE_RS_EVENTS__EMPTY_CYCLES = 0x53015E;
  static constexpr unsigned int CODE_RS_EVENTS__EMPTY_END  = 0x1D7015E;
  static constexpr unsigned int CODE_UOPS_EXECUTED__CYCLES_GE_1_UOP_EXEC = 0x15301B1;
  static constexpr unsigned int CODE_UOPS_EXECUTED__CYCLES_GE_2_UOPS_EXEC = 0x25301B1;
  static constexpr unsigned int CODE_UOPS_EXECUTED__THREAD = 0x5301B1;
  static constexpr unsigned int CODE_UOPS_ISSUED__ANY = 0x53010E;
  static constexpr unsigned int CODE_UOPS_RETIRED__RETIRE_SLOTS = 0x5302C2;

  static constexpr int PipelineWidth = 4;
  static constexpr int MEM_L3_WEIGHT = 7;

  Type types[NGROUPS][METRIC_COUNT] = {
    {
      PERF_TYPE_HARDWARE, // PERF_COUNT_HW_CPU_CYCLES
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_CPU_CLOCK
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_TASK_CLOCK
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    },
   {
      PERF_TYPE_HARDWARE, // PERF_COUNT_HW_CPU_CYCLES
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_CPU_CLOCK
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_TASK_CLOCK
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    },
   {
      PERF_TYPE_HARDWARE, // PERF_COUNT_HW_CPU_CYCLES
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_CPU_CLOCK
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_TASK_CLOCK
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW,
      PERF_TYPE_RAW
    },
   {
      PERF_TYPE_HARDWARE, // PERF_COUNT_HW_CPU_CYCLES
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_CPU_CLOCK
      PERF_TYPE_SOFTWARE, // PERF_COUNT_SW_TASK_CLOCK
      PERF_TYPE_RAW,
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

      CODE_INST_RETIRED__ANY,
      CODE_IDQ_UOPS_NOT_DELIVERED__CORE,
      CODE_CYCLE_ACTIVITY__STALLS_LDM_PENDING,
      CODE_RESOURCE_STALLS__SB
     },
    {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,

      CODE_UOPS_ISSUED__ANY,
      CODE_UOPS_RETIRED__RETIRE_SLOTS,
      CODE_INT_MISC__RECOVERY_CYCLES,
      CODE_ARITH__FPU_DIV_ACTIVE

    },
    {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,

      CODE_UOPS_EXECUTED__CYCLES_GE_1_UOP_EXEC,
      CODE_UOPS_EXECUTED__CYCLES_GE_2_UOPS_EXEC,
      CODE_RS_EVENTS__EMPTY_CYCLES,
      CODE_CYCLE_ACTIVITY__CYCLES_NO_EXECUTE
    },
    {
      PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_SW_CPU_CLOCK,
      PERF_COUNT_SW_TASK_CLOCK,

      CODE_MEM_LOAD_UOPS_RETIRED__LLC_HIT,
      CODE_MEM_LOAD_UOPS_RETIRED__LLC_MISS, 
      CODE_CYCLE_ACTIVITY__STALLS_L2_PENDING,
      CODE_ICACHE__IFETCH_STALL    
      //      CODE_IDQ_UOPS_NOT_DELIVERED__CYCLES_0_UOPS_DELIV__CORE
    }
  };



  TopDown(bool imultiplex=false) : PerfStatBase<4>(imultiplex){init();}
  TopDown(PerfStatBase::FD f) : PerfStatBase<4>(f){}

  void get(Conf * c, Type * t) const {
    memcpy(c,&confs[0][0], NGROUPS*METRIC_COUNT*sizeof(Conf));
    memcpy(t,&types[0][0], NGROUPS*METRIC_COUNT*sizeof(Type));

  };



  double CYCLES(int n) const { return double(results[n][METRIC_OFFSET+0]);}
  double SLOTS(int n) const { return PipelineWidth*CYCLES(n);}

  long long INST_RETIRED__ANY() const { return results[0][METRIC_OFFSET+3];}

  // backward compatible interface....
  long long instructionsRaw() const { return INST_RETIRED__ANY();}
  double instructionsTot() const { return double(INST_RETIRED__ANY())*(CYCLES(0)+CYCLES(1)+CYCLES(2)+CYCLES(3))/CYCLES(0);}
  double instructions() const { return (0==calls()) ? 0 : instructionsTot()/double(calls()); }


   // instructions per cycle
  double ipc() const { return double(INST_RETIRED__ANY())/CYCLES(0);}

  // raw

  long long IDQ_UOPS_NOT_DELIVERED__CORE() const { return results[0][METRIC_OFFSET+4];}
  long long CYCLE_ACTIVITY__STALLS_LDM_PENDING() const { return results[0][METRIC_OFFSET+5];}
  long long RESOURCE_STALLS__SB() const { return results[0][METRIC_OFFSET+6];}


  long long UOPS_ISSUED__ANY()  const { return results[1][METRIC_OFFSET+3];}
  long long UOPS_RETIRED__RETIRE_SLOTS()  const { return results[1][METRIC_OFFSET+4];}
  long long INT_MISC__RECOVERY_CYCLES()  const { return results[1][METRIC_OFFSET+5];}
  long long ARITH__FPU_DIV_ACTIVE() const { return results[1][METRIC_OFFSET+6];}

  long long UOPS_EXECUTED__CYCLES_GE_1_UOP_EXEC() const { return results[2][METRIC_OFFSET+3];}
  long long UOPS_EXECUTED__CYCLES_GE_2_UOPS_EXEC() const { return results[2][METRIC_OFFSET+4];}
  long long RS_EVENTS__EMPTY_CYCLES() const { return results[2][METRIC_OFFSET+5];}
  long long CYCLE_ACTIVITY__CYCLES_NO_EXECUTE() const { return results[2][METRIC_OFFSET+6];}



  long long MEM_LOAD_UOPS_RETIRED__LLC_HIT()  const { return results[3][METRIC_OFFSET+3];}
  long long MEM_LOAD_UOPS_RETIRED__LLC_MISS()  const { return results[3][METRIC_OFFSET+4];}
  long long CYCLE_ACTIVITY__STALLS_L2_PENDING()  const { return results[3][METRIC_OFFSET+5];}
  long long IDQ_UOPS_NOT_DELIVERED__CYCLES_0_UOPS_DELIV__CORE() const  { return results[3][METRIC_OFFSET+6];}
  long long ICACHE__IFETCH_STALL() const  { return results[3][METRIC_OFFSET+6];}


  // level 1
  double frontendBound() const { return IDQ_UOPS_NOT_DELIVERED__CORE() / SLOTS(0);}
  double backendBound() const { return 1. - ( frontendBound() + badSpeculation() + retiring() ); } 
  double badSpeculation() const { 
    return ( UOPS_ISSUED__ANY() - UOPS_RETIRED__RETIRE_SLOTS() + 
	     PipelineWidth *  INT_MISC__RECOVERY_CYCLES() ) / SLOTS(1);
  }
  double retiring() const {
    return UOPS_RETIRED__RETIRE_SLOTS() / SLOTS(1);
  }

  // level2

  double frontLatency() const { 
    return IDQ_UOPS_NOT_DELIVERED__CYCLES_0_UOPS_DELIV__CORE()/CYCLES(3);
  }

  double iCache() const { 
    return ICACHE__IFETCH_STALL()/CYCLES(3);
  }


  double backendBoundAtEXE_stalls() const {
    return CYCLE_ACTIVITY__CYCLES_NO_EXECUTE() + UOPS_EXECUTED__CYCLES_GE_1_UOP_EXEC() 
      - UOPS_EXECUTED__CYCLES_GE_2_UOPS_EXEC() - RS_EVENTS__EMPTY_CYCLES();
  }

  double backendBoundAtEXE() const { return backendBoundAtEXE_stalls()/CYCLES(2);} 

  double memBoundFraction() const {
    return double( CYCLE_ACTIVITY__STALLS_LDM_PENDING() + RESOURCE_STALLS__SB() ) 
      / double( backendBoundAtEXE_stalls()*(CYCLES(0)/CYCLES(2)) + RESOURCE_STALLS__SB() );
  }

  double memBound() const {
    return  memBoundFraction()*backendBoundAtEXE();
  }
  
  
  double coreBound() const {
    return backendBoundAtEXE() - memBound();
  }


  // level3
  
  double memL3HitFraction() const { return 
      double(  MEM_LOAD_UOPS_RETIRED__LLC_HIT()) / 
      double ( MEM_LOAD_UOPS_RETIRED__LLC_HIT() + MEM_L3_WEIGHT * MEM_LOAD_UOPS_RETIRED__LLC_MISS());
  } 
  
  double memL3Bound() const { return memL3HitFraction() * CYCLE_ACTIVITY__STALLS_L2_PENDING() / CYCLES(3);}
  double dramBound() const { return (1.-memL3HitFraction()) * CYCLE_ACTIVITY__STALLS_L2_PENDING() / CYCLES(3);}
  
  
  double divideBound() const {
    return ARITH__FPU_DIV_ACTIVE()/CYCLES(1);
  }
  
  
  void header(std::ostream & out, bool details=false) const {
    const char * sepF = "|  *"; 
    const char * sep = "*|  *"; 
    const char * sepL = "*|"; 
    out << sepF << "real time"
        << sep << "task time"
   	<< sep << "cycles" 
	<< sep << "ipc"
      
   	<< sep << "frontend" 
   	<< sep << "backend" 
   	<< sep << "bad spec" 
   	<< sep << "retiring" 
      
      // << sep << "front lat" 
  	<< sep << "icache" 
      
  	<< sep << "exe" 
   	<< sep << "mem" 
   	<< sep << "core" 
      
	<< sep << "l3/cy"
	<< sep << "dram/cy"
      
	<< sep << "div/cy"
      
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
      
	<< sep << percent*frontendBound()
	<< sep << percent*backendBound()
	<< sep << percent*badSpeculation()
	<< sep << percent*retiring()
      
      //	<< sep << percent*frontLatency()
	<< sep << percent*iCache()      

      	<< sep << percent*backendBoundAtEXE()
	<< sep << percent*memBound()
	<< sep << percent*coreBound()
      
	<< sep << percent*memL3Bound()
	<< sep << percent*dramBound()
      
	<< sep << percent*divideBound()
      
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
