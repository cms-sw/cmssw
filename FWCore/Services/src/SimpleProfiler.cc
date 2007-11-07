
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309
#endif

#include <cstdio>

#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <sys/time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <ucontext.h>
#include <execinfo.h>

#include "SimpleProfiler.h"
#include "ProfParse.h"

#ifndef __USE_POSIX199309
#error "SimpleProfile requires the definition of __USE_POSIX199309"
#endif


namespace INSTR
{
  typedef unsigned char byte;
  const byte RET = 0xc3;  
}

namespace
{

  std::string makeFileName()
  {
    pid_t p = getpid();
    std::ostringstream ost;
    ost << "profdata_" << p;
    return ost.str();
  }
}

// ---------------------------------------------------------------------
// Macros, for this compilation unit only
// ---------------------------------------------------------------------

#define MUST_BE_ZERO(fun) if((fun)!=0) { perror("function failed"); abort(); }
#define getBP(X)  asm ( "movl %%ebp,%0" : "=m" (X) )
#define getSP(X)  asm ( "movl %%esp,%0" : "=m" (X) )

#if 0
#define DODEBUG if(1) std::cerr
#else
#define DODEBUG if(0) std::cerr
#endif



#ifndef __USE_GNU
const int REG_EIP = 14;
const int REG_EBP = 6;
const int REG_ESP = 7;
#endif

#include "unistd.h"
#include <sstream>
#include <fstream>


namespace 
{

  struct frame
  {
    // member data
    frame* next;
    void*  ip;

    // member functions
    frame() : next(0), ip(0) { }
    unsigned char instruction() const { return *(unsigned char*)ip; }
    void print(FILE* out) { fprintf(out, "next: %8.8x  ip: %8.8x  inst: %2.2x\n", next, ip, instruction()); }
  };

  void write_maps()
  {
    pid_t pid = getpid();
    std::ostringstream strinput,stroutput;
    strinput << "/proc/" << pid << "/maps";
    stroutput << "profdata_" << pid << "_maps";
    std::ifstream input(strinput.str().c_str());
    std::ofstream output(stroutput.str().c_str());
    std::string line;
    while (getline(input, line)) output << line << '\n';
    input.close();
    output.close();
  }

  FILE* frame_cond = 0;

  void openCondFile()
  {
    std::string filename(makeFileName());
    filename += "_condfile";
    frame_cond = fopen(filename.c_str(),"w");
    if(frame_cond==0)
      {
	std::cerr << "bad open of profdata_condfile\n";
	throw std::runtime_error("bad open");
      }
  }

  void writeCond(int code)
  {
    fwrite(&code,sizeof(int),1,frame_cond);
  }

  void closeCondFile()
  {
    fclose(frame_cond);
  }

  void dumpStack(const char* msg,
		 unsigned int* esp, unsigned int* ebp, unsigned char* eip,
		 ucontext_t* ucp)
  {
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
    throw logic_error("Cannot dumpStack on 64 bit build");
#else
    fprintf(frame_cond, msg);
    fflush(frame_cond);
    return;
    fprintf(frame_cond, "dumpStack:\n i= %x\n eip[0]= %2.2x\nb= %x\n s= %x\n b[0]= %x\n b[1]= %x\n b[2]= %x\n",
	    (void*)eip, eip[0], (void*)ebp, (void*)esp, (void*)(ebp[0]), (void*)(ebp[1]), (void*)(ebp[2]));
    fflush(frame_cond);


#if 0
    unsigned int* spp = esp;
    for(int i=15;i>-5;--i)
      {
	fprintf(frame_cond, "    %x esp[%d]= %x\n", (void*)(spp+i), i, (void*)*(spp+i));
	fflush(frame_cond);
      }
#else
    while(ucp->uc_link)
      {
	fprintf(frame_cond, "   %8.8x\n",ucp->uc_link);
	ucp = ucp->uc_link;
      }
#endif
#endif
  }
}

namespace {
  // counters
  int samples_total=0;
  int samples_uninterpretable=0;
  int samples_ebp_less_esp=0;
  int samples_bad_ebp=0;
  int samples_after_push=0;
  int samples_before_push=0;
  int samples_between_leave_ret=0;
  int samples_skipped=0;
  int samples_outside_range=0;
  int samples_popped=0;
}

// ---------------------------------------------------------------------
// The signal handler. We register this to handle the SIGPROF signal.
// ---------------------------------------------------------------------

extern "C" 
{
  void sigFunc(int sig, siginfo_t* info, void* context)
  {
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
    return;
#else
    int condition = 0;
    //fprintf(stderr, "--------------------\n");
    unsigned int* this_sp;
    unsigned int* this_bp;
    getSP(this_sp);
    getBP(this_bp);

    DODEBUG << "got the interrupt: " << sig << " " << pthread_self() << "\n";
    SimpleProfiler* prof = SimpleProfiler::instance();
    ucontext_t* ucp = (ucontext_t*)context;
    unsigned char* eip=(unsigned char*)ucp->uc_mcontext.gregs[REG_EIP];
    void* stacktop = prof->stackTop();
    void** arr = prof->tempStack();
    ++samples_total;

#if 0
    std::cerr << "siginfo=" << (void*)info
	 << " context=" << context
	 << " stack=" << (void*)ucp->uc_stack.ss_sp
	 << "\n";
#endif

    // ----- manual way ------
    unsigned int* ebp=(unsigned int*)ucp->uc_mcontext.gregs[REG_EBP];
    unsigned int* esp=(unsigned int*)ucp->uc_mcontext.gregs[REG_ESP];
    int* fir = (int*)arr;
    int* cur = fir;
    bool stack_uninterpretable = false;

    if (ebp<esp)
      {
	//++error_count;
	getBP(ebp);
	if(ebp<esp)
	  {
	    *cur++ = ((unsigned int)eip);
	    *cur++ = 0U;
	    //std::cerr << "early completion for eip = " << (unsigned int)eip << '\n';
	    fprintf(frame_cond," early completion for eip %8.8x\n",(unsigned int)eip);

	    ++samples_uninterpretable;
	    stack_uninterpretable=true;
	  }
	else
	  {
	    fprintf(frame_cond,"ebp < esp (but not early completion)\n");
	    ebp=(unsigned int*)(*ebp);
	    ebp=(unsigned int*)(*ebp);
	    ++samples_ebp_less_esp;
	  } 
      }
  
    if (!stack_uninterpretable)
      {
	*cur++ = ((unsigned int)eip);

	// Now we handle special cases:

	//  ... if we've done the LEAVE, but not yet the RET, then
	//  we've already popped the frame for the function we're
	//  in. We then need to capture the calling routine *from the
	//  top of the stack*, and then can continue with the normal
	//  walk up the stack.
	//writeCond(*(esp-1));
	//writeCond((int)ebp);

	if(eip[0]==INSTR::RET)
	  {
	    ++samples_between_leave_ret;
	    //std::cerr << "after the leave and immediately before the ret\n";
	    condition += 1;
	    //*cur++ = ((unsigned int)*esp);
	    *cur++ = *esp;
	  }

	// ... when entering a function, we encounter the following
	// instructions:
	//
	//       PUSH   %ebp
	//       MOV    %esp,%ebp
	//
	// If we're in the new function, but before these instructions
	// have been done, then EBP and ESP aren't yet set for the new
	// function... so set them.

	if(
	   (esp)<(unsigned int*)0x0000ffff||
	   (esp+1)<(unsigned int*)0x0000ffff||
	   (esp+2)<(unsigned int*)0x0000ffff
	   )
	  {
	    dumpStack("bad ebp data ****************\n",esp,ebp,eip,ucp); 
	  }
	if( *esp==(unsigned int)ebp )
	  {
	    ++samples_after_push;
	    condition += 4;	    
	    //std::cerr << "after the push, before the mov\n";
	    ebp = esp;
	  }
	else if(eip[0]==0x55 && eip[1]==0x89 && eip[2]==0xe5)
	  {
	    ++samples_before_push;
	    condition += 8;
	    //std::cerr << "after the call, before the push\n";
	    *cur++ = *esp;
	  }
	// if value at stacktop and stacktop-1 are outside stack range
	else if((*(esp-2)==0xadb8 && *(esp-1)==0x80cd00))
	  if( (unsigned int)eip!=*(esp-1) && *eip!=0xc3 && *eip!=0xe8 )
	    {
	      ++samples_outside_range;
	      dumpStack("stacktop and stacktop-1 outside range\n",
			esp,ebp,eip,ucp);
	      
	      if ( (void*)eip > stacktop)
		{
		  // The current function had no stack frame.
		  ++samples_bad_ebp;
		  condition += 16;
		  if( (ebp+0)<(unsigned int*)0x0000ffff )
		    {
		      dumpStack("bad ebp data ****************\n",
				esp,ebp,eip,ucp);
		    }
		  ebp = (unsigned int*)esp[0];
                  ebp=0; // we will skip these types of frames for now!!!! (jbk)

		  // 		frame* f = (frame*)(esp[0]);
		  // 		while ( f != 0 )
		  // 		  {
		  // 		    f->print(frame_cond);
		  // 		    f = f->next;
		  // 		  }
		  
		}
	      else
		{
		  // The stack frame for the current function has
		  // already been popped.
		  ++samples_popped;
		  condition += 2;
		  //std::cerr << "after the leave but before the ret\n";
		  *cur++ = *esp;
		}
	    }
	  else
	    {
	      //std::cerr << "skip stack trace\n";
	      ++samples_skipped;
	      ebp=0;
	    }
	
	if (ebp<reinterpret_cast<unsigned int*>(stacktop) == false) 
	  fprintf(stderr, "--- not going through the loop this time\n");

	//if (condition!=0) fprintf(stderr, "bad condition: %d\n", condition);
	//while(ebp<stacktop)

	int counter = 0;
	while (ebp && ebp<stacktop)
	  {
	    //*cur++   = *(ebp+1);
	    if( (ebp+1)<(unsigned int*)0x0000ffff )
	      {
		fprintf(frame_cond,"bad ebp+1 data %8.8x\n",(ebp+1));
		//dumpStack("bad ebp+1 data ****************\n",
		//	  esp,ebp,eip,ucp); 
		break;
	      }
	    unsigned int ival = ebp[1];
	    *cur = ival;
	    cur++;

	    //ebp=(unsigned int*)(*ebp);
	    if( (ebp)<(unsigned int*)0x0000ffff )
	      {
		fprintf(frame_cond,"bad ebp data\n");
		//dumpStack("bad ebp data ****************\n",
		//	  esp,ebp,eip,ucp); 
		break;
	      }
	    unsigned int val = ebp[0];
	    ebp = reinterpret_cast<unsigned int*>(val);

	    if (++counter > 1000)
	      {
		std::cerr << "BAD COUNT!!!!!!\n";
		break;
	      }
	  }
      }
    //writeCond(condition)
    prof->commitFrame((void**)fir,(void**)cur);    
#endif
  }
}

namespace
{
  
  stack_t ss_area;

  void setupTimer();

  void* setStacktop()
  {
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
    throw logic_error("setStacktop not callable on 64 bit platform");
    return 0;
#else
    const std::string target_name("__libc_start_main");
    unsigned int* ebp_reg;
    getBP(ebp_reg);
    unsigned int* ebp = (unsigned int*)(ebp_reg);
    unsigned int* top = (unsigned int*)(ebp);

    void* address_of_target = 0;
    const int max_stackdepth=30;
    void* sta[max_stackdepth];

    int depth = backtrace(sta, max_stackdepth);
    int cnt = depth;
    if (depth > 1) depth-=1;

    // find function one below main()
    Dl_info look;
    for (int i=0; i<cnt && !address_of_target; ++i)
      {
	if(dladdr(sta[i],&look)!=0)
	  {
#if 0
	    std::cerr << "sta[" << i << "]=" << sta[i] << ":  " << (look.dli_saddr ? look.dli_sname : "?") << ":" << look.dli_saddr << "\n"; 
#endif
	    if (look.dli_saddr && target_name==look.dli_sname)
	      {
		address_of_target = sta[i];
		// std::cerr << "found " << target_name << "\n";
	      }
	  }
	else
	  {
	    // This isn't really an error; it just means the function
	    // was not found by the dynamic loader. The function might
	    // be one that is declared 'static', and is thus not
	    // visible outside of its compilation unit.
	    std::cerr << "setStacktop: no function information for " 
		 << sta[i] 
		 << "\n";
	  }
      }

    if (address_of_target == 0)
      throw std::runtime_error("no main function found in stack");

    //fprintf(stderr,"depth=%d top=%8.8x\n",depth,top);

    // Now we walk toward the beginning of the stack until we find the
    // frame that is holding the return address of our target function.

    // depth is how many more frames there are to look at in the stack.
    // top is the frame we are currently looking at.
    // top+1 is the address to which the current frame will return.
    while (depth>0 && (void*)*(top+1) != address_of_target)
    {
      //fprintf(stderr,"depth=%d top=%8.8x func=%8.8x\n",depth,top,*(top+1));
      if (top<(unsigned int*)0x10) fprintf(stderr,"problem\n");
      top=(unsigned int*)(*top);
      --depth;
    };

    if (depth==0) 
      throw std::runtime_error("setStacktop: could not find stack bottom");

    // Now we have to move one frame more, to the frame of the target
    // function. We want the location in memory of this frame (not any
    // address stored in the frame, but the address of the frame
    // itself).
    top=(unsigned int*)(*top);

    return top;
#endif
  }

  void setupTimer()
  {
#if USE_SIGALTSTACK
    static vector<char> charbuffer(SIGSTKSZ);
    //ss_area.ss_sp = new char[SIGSTKSZ];
    ss_area.ss_sp = &charbuffer[0];
    ss_area.ss_flags = 0;
    ss_area.ss_size = SIGSTKSZ;
#endif
    //static int is_set = 0;
    //if(is_set!=1) { ++is_set; return; }
    //else ++is_set;

    sigset_t myset,oldset;
    // all blocked for now
    MUST_BE_ZERO(sigfillset(&myset));
    MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&myset,&oldset));

#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
    // ignore all the RT signals
    struct sigaction tmpact;
    memset(&tmpact,0,sizeof(tmpact));
    tmpact.sa_handler = SIG_IGN;

    for(int num=SIGRTMIN;num<SIGRTMAX;++num)
      {
	MUST_BE_ZERO(sigaddset(&oldset,num));
	MUST_BE_ZERO(sigaction(num,&tmpact,NULL));
      }
#endif

#if USE_SIGALTSTACK
    if(sigaltstack(&ss_area,0)!=0)
      {
	perror("sigaltstack failed for profile timer interrupt");
	abort();
      }
#endif

    // set up my RT signal now
    struct sigaction act;
    memset(&act,0,sizeof(act));
    act.sa_sigaction = sigFunc;
    act.sa_flags = SA_RESTART | SA_SIGINFO | SA_ONSTACK;

    // get my signal number
    int mysig = SIGPROF;

    if(sigaction(mysig,&act,NULL)!=0)
      {
	perror("sigaction failed");
	abort();
      }

    // Turn off handling of SIGSEGV signal
    memset(&act,0,sizeof(act));
    act.sa_handler = SIG_DFL;

    if (sigaction(SIGSEGV, &act, NULL) != 0)
      {
	perror("sigaction failed");
	abort();
      }

    struct rlimit limits;
    if (getrlimit(RLIMIT_CORE, &limits) != 0)
      {
	perror("getrlimit failed");
	abort();
      }
    std::cerr << "core size limit (soft): " << limits.rlim_cur << '\n';
    std::cerr << "core size limit (hard): " << limits.rlim_max << '\n';

    struct itimerval newval;
    struct itimerval oldval;

    newval.it_interval.tv_sec  = 0;
    newval.it_interval.tv_usec = 10000;
    newval.it_value.tv_sec  = 0;
    newval.it_value.tv_usec = 10000;

    if(setitimer(ITIMER_PROF,&newval,&oldval)!=0)
      {
	perror("setitimer failed");
	abort();
      }

    // reenable the signals, including my interval timer
    MUST_BE_ZERO(sigdelset(&oldset,mysig));
    MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&oldset,0));
  }


  struct AdjustSigs
  {
    AdjustSigs()
    {
      sigset_t myset,oldset;
      // all blocked for now
      MUST_BE_ZERO(sigfillset(&myset));
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&myset,&oldset));
    
      // ignore all the RT signals
      struct sigaction tmpact;
      memset(&tmpact,0,sizeof(tmpact));
      tmpact.sa_handler = SIG_IGN;
    
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
      for(int num=SIGRTMIN;num<SIGRTMAX;++num)
	{
	  MUST_BE_ZERO(sigaddset(&oldset,num));
	  MUST_BE_ZERO(sigaction(num,&tmpact,NULL));
	}
#endif
    
      MUST_BE_ZERO(sigaddset(&oldset,SIGPROF));
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&oldset,0));
    }
  };

  static AdjustSigs global_adjust;


}

SimpleProfiler* SimpleProfiler::inst_ = 0;
boost::mutex SimpleProfiler::lock_;


SimpleProfiler* SimpleProfiler::instance()
{
  if(SimpleProfiler::inst_ == 0)
    {
      boost::mutex::scoped_lock sl(lock_);
      if(SimpleProfiler::inst_ == 0)
	{
	  static SimpleProfiler p;
	  SimpleProfiler::inst_ = &p;
	}
    }
  return SimpleProfiler::inst_;
}

SimpleProfiler::SimpleProfiler():
  frame_data_(10*1000*1000),
  tmp_stack_(1000),
  high_water_(&frame_data_[10*1000*1000-10*1000]),
  curr_(&frame_data_[0]),
  filename_(makeFileName()),
  fd_(open(filename_.c_str(), 
	   O_RDWR|O_CREAT,
	   S_IRWXU|S_IRGRP|S_IROTH|S_IWGRP|S_IWOTH)),
  installed_(false),
  running_(false),
  owner_(),
  stacktop_(setStacktop())
{
  if (fd_ < 0) {
      std::ostringstream ost;
      ost << "failed to open profiler output file " << filename_;
      throw std::runtime_error(ost.str().c_str());
  }
  
  openCondFile();
}

SimpleProfiler::~SimpleProfiler()
{
  if (running_)
    std::cerr << "Warning: the profile timer was not stopped,\n"
	 << "profiling data in " << filename_
	 << " is probably incomplete and will not be processed\n";

  closeCondFile();
}

void SimpleProfiler::commitFrame(void** first, void** last)
{
  void** cnt_ptr = curr_; 
  *cnt_ptr = reinterpret_cast<void*>(std::distance(first,last));
  ++curr_;
  curr_ = std::copy(first,last,curr_);
  if(curr_ > high_water_) doWrite();
}

void SimpleProfiler::doWrite()
{
  void** start = &frame_data_[0];
  if(curr_ == start) return;
  unsigned int cnt = std::distance(start,curr_) * sizeof(void*);
  unsigned int totwr=0;

  while (cnt>0 && (totwr=write(fd_,start,cnt)) != cnt) {
      if(totwr==0) 
	throw std::runtime_error("SimpleProfiler::doWrite wrote zero bytes");
      start+=(totwr/sizeof(void*));
      cnt-=totwr;
  }

  curr_ = &frame_data_[0];
}

void SimpleProfiler::start()
{
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
  throw logic_error("SimpleProfiler not available on 64 bit platform");
#endif
  {
    boost::mutex::scoped_lock sl(lock_);

    if (installed_) {
	std::cerr << "Warning: second thread " << pthread_self()
	     << " requested the profiler timer and another thread\n"
	     << owner_ << "has already started it.\n"
	     << "Only one thread can do profiling at a time.\n"
	     << "This second thread will not be profiled.\n";
	return;
    }
    
    installed_ = true;
  }

  owner_ = pthread_self();
  setupTimer();
  running_ = true;
}

void SimpleProfiler::stop()
{
  if(!installed_)
    {
      std::cerr << "SimpleProfiler::stop - no timer installed to stop\n";
      return;
    }

  if(owner_ != pthread_self())
    {
      std::cerr << "SimpleProfiler::stop - only owning thread can stop the timer\n";
      return;
    }

  if(!running_)
    {
      std::cerr << "SimpleProfiler::stop - no timer is running\n";
      return;
    }

  struct itimerval newval;

  newval.it_interval.tv_sec  = 0;
  newval.it_interval.tv_usec = 0;
  newval.it_value.tv_sec  = 0;
  newval.it_value.tv_usec = 0;

  if(setitimer(ITIMER_PROF,&newval,0)!=0)
    {
      perror("setitimer call failed - could not stop the timer");
    }

  running_=false;
  complete();
}



void SimpleProfiler::complete()
{
  doWrite();

  if(lseek(fd_,0,SEEK_SET)<0)
    {
      std::cerr << "SimpleProfiler: could not seek to the start of the profile\n"
	   << " data file during completion.  Data will be lost.\n";
      return;
    }

  writeProfileData(fd_,filename_);
  write_maps();

  std::string totsname = makeFileName();
  totsname += "_sample_info";
  std::ofstream ost(totsname.c_str());
  
  ost << "samples_total " << samples_total << "\n"
      << "samples_uninterpretable " << samples_uninterpretable << "\n"
      << "samples_ebp_less_esp " << samples_ebp_less_esp << "\n"
      << "samples_bad_ebp " << samples_bad_ebp << "\n"
      << "samples_after_push " << samples_after_push << "\n"
      << "samples_before_push " << samples_before_push << "\n"
      << "samples_between_leave_ret " << samples_between_leave_ret << "\n"
      << "samples_skipped " << samples_skipped << "\n"
      << "samples_outside_range " << samples_outside_range << "\n"
      << "samples_popped " << samples_popped << "\n"
    ;
}



// -----------------------------------------------

