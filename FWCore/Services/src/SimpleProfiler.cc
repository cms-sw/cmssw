
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309
#endif


#include <csignal>
#include <cerrno>
#include <cstdio>
#include <ctime>

#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ucontext.h>
#include <execinfo.h>

#include "SimpleProfiler.h"

#ifndef __USE_POSIX199309
#error "bad - did not define __USE_POSIX199309"
#endif

using namespace std;

extern void writeProfileData(int fd, const std::string& prefix);

stack_t ss_area;

// --------------------------

void setupTimer();
unsigned int* setStacktop();

SimpleProfiler* SimpleProfiler::inst_ = 0;
boost::mutex SimpleProfiler::lock_;

string makeFileName()
{
  pid_t p = getpid();
  ostringstream ost;
  ost << "profdata_" << p;
  return ost.str();
}

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
  frame_data_(1000000*10),
  tmp_stack_(1000),
  high_water_(&frame_data_[1000000*10-10000]),
  curr_(&frame_data_[0]),
  filename_(makeFileName()),
  fd_(open(filename_.c_str(),O_RDWR|O_CREAT,
	   S_IRWXU|S_IRGRP|S_IROTH|S_IWGRP|S_IWOTH)),
  installed_(false),
  running_(false),
  owner_(),
  stacktop_(setStacktop())
{
  if(fd_<0)
    {
      ostringstream ost;
      ost << "failed to open profiler output file " << filename_;
      throw runtime_error(ost.str().c_str());
    }
}

SimpleProfiler::~SimpleProfiler()
{
  if(running_)
    cerr << "Warning: the profile timer was not stopped,\n"
	 << "profiling data in " << filename_
	 << " is probably incomplete and will not be processed\n";
}

void SimpleProfiler::commitFrame(void** first, void** last)
{
  unsigned int* cnt_ptr = (unsigned int*)curr_; 
  *cnt_ptr = distance(first,last);
  ++curr_;
  curr_ = copy(first,last,curr_);
  if(curr_ > high_water_) doWrite();
}

void SimpleProfiler::doWrite()
{
  void** start = &frame_data_[0];
  if(curr_ == start) return;
  unsigned int cnt = distance(start,curr_) * sizeof(void*);
  unsigned int totwr=0;

  while(cnt>0 && (totwr=write(fd_,start,cnt))!=cnt)
    {
      if(totwr==0)
	throw runtime_error("SimpleProfiler::doWrite wrote zero bytes");
      start+=(totwr/sizeof(void*));
      cnt-=totwr;
    }

  curr_ = &frame_data_[0];
}

void SimpleProfiler::start()
{
  {
    boost::mutex::scoped_lock sl(lock_);

    if(installed_)
      {
	cerr << "Warning: second thread " << pthread_self()
	     << " requested the profiler timer and another thread\n"
	     << owner_ << "has already started it.\n"
	     << "Only one thread can do profiling at a time.\n";
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
      cerr << "SimpleProfiler::stop - no timer installed to stop\n";
      return;
    }

  if(owner_ != pthread_self())
    {
      cerr << "SimpleProfiler::stop - only owning thread can stop the timer\n";
      return;
    }

  if(!running_)
    {
      cerr << "SimpleProfiler::stop - no timer is running\n";
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

// -------------------------------------

#define MUST_BE_ZERO(fun) if((fun)!=0)		\
    { perror("function failed"); abort(); }

#define getBP(X)  asm ( "movl %%ebp,%0" : "=r" (X) )

#if 0
#define DODEBUG if(1) cerr
#else
#define DODEBUG if(0) cerr
#endif

#ifndef __USE_GNU
    const int REG_EIP = 14;
    const int REG_EBP = 6;
    const int REG_ESP = 7;
#endif

extern "C" {
  void sigFunc(int sig, siginfo_t* info, void* context)
  {

#if 0   
    if(rand()%50==2)
      {
	char* crasher = (char*)0x83339999;
	char badone = *crasher;
      }
#endif 

    DODEBUG << "got the interrupt: " << sig << " " << pthread_self() << "\n";
    SimpleProfiler* prof = SimpleProfiler::instance();
    ucontext_t* ucp = (ucontext_t*)context;
    unsigned char* eip=(unsigned char*)ucp->uc_mcontext.gregs[REG_EIP];
    unsigned int* stacktop = prof->stackTop();
    void** arr = prof->tempStack();

#if 0
    int cnt = backtrace(arr,1000);

    // arr[0] = this function
    // arr[1] = bridge function into ISR
    // arr[2] = first useful one
    // arr[cnt-2] -> usually zero
    // arr[cnt-1] -> usually not useful
    // arr[1] -> replace with address at eip

    arr[0]=eip;
    prof->commitFrame(&arr[0],&arr[cnt-2]);
#endif
#if 0
    void** cur = &arr[0];
    cerr << (void*)ucp->uc_link << " " << (int)(cur-&arr[0]) << "-----\n";
    eip=(unsigned char*)ucp->uc_mcontext.gregs[REG_EIP];
    *cur++ = eip;
    cerr << "ss_sp=" << ucp->uc_stack.ss_sp << "\n"
	 << "ss_flags=" << ucp->uc_stack.ss_flags << "\n"
	 << "ss_size=" << ucp->uc_stack.ss_size << "\n";
    prof->commitFrame(&arr[0],cur);
    
#endif

#if 1
    // ----- manual way ------
    unsigned int* ebp=(unsigned int*)ucp->uc_mcontext.gregs[REG_EBP];
    unsigned int* esp=(unsigned int*)ucp->uc_mcontext.gregs[REG_ESP];
    int* fir = (int*)arr;
    int* cur = fir;
    bool done = false;

    if(ebp<esp)
      {
	//++error_count;
	getBP(ebp);
	if(ebp<esp)
	  {
	    *cur++ = ((unsigned int)eip);
	    *cur++ = ((unsigned int)ebp);
	    //cerr << "early completion\n";
	    done=true;
	  }
	else
	  {
	    ebp=(unsigned int*)(*ebp);
	    ebp=(unsigned int*)(*ebp);
	  }
      }
  
    if(!done)
      {
      
	if(eip[0]==0xc3)
	  {
	    //cerr << "after the leave but before the ret\n";
	    *cur++ = ((unsigned int)*esp);
	  }
	else
	  *cur++ = ((unsigned int)eip);

	if( *esp==(unsigned int)ebp )
	  {
	    //cerr << "we are after the push, before the mov\n";
	    ebp = esp;
	  }
	else if(eip[0]==0x55 && eip[1]==0x89 && eip[2]==0xe5)
	  {
	    //cerr << "before the push, after the call\n";
	    *cur++ = ((unsigned int)*esp);
	  }
      
	while(ebp<stacktop)
	  {
	    //fprintf(stderr,"loop %8.8x %8.8x\n",ebp,*(ebp+1));	  
	    *cur++ = (*(ebp+1));
	    ebp=(unsigned int*)(*ebp);
	  }
	//++total_sample_count;
      }

    prof->commitFrame((void**)fir,(void**)cur);    
#endif
  }
}

void SimpleProfiler::complete()
{
  doWrite();

  if(lseek(fd_,0,SEEK_SET)<0)
    {
      cerr << "SimpleProfiler: could not seek to the start of the profile\n"
	   << " data file during completion.  Data will be lost.\n";
      return;
    }

  writeProfileData(fd_,filename_);
}

void setupTimer()
{
  ss_area.ss_sp = new char[SIGSTKSZ];
  ss_area.ss_flags = 0;
  ss_area.ss_size = SIGSTKSZ;

  //static int is_set = 0;
  //if(is_set!=1) { ++is_set; return; }
  //else ++is_set;

  sigset_t myset,oldset;
  // all blocked for now
  MUST_BE_ZERO(sigfillset(&myset));
  MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&myset,&oldset));

  // ignore all the RT signals
  struct sigaction tmpact;
  memset(&tmpact,0,sizeof(tmpact));
  tmpact.sa_handler = SIG_IGN;

  for(int num=SIGRTMIN;num<SIGRTMAX;++num)
    {
      MUST_BE_ZERO(sigaddset(&oldset,num));
      MUST_BE_ZERO(sigaction(num,&tmpact,NULL));
    }

  if(sigaltstack(&ss_area,0)!=0)
    {
      perror("sigaltstack failed for profile timer interrupt");
      abort();
    }

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
    
    for(int num=SIGRTMIN;num<SIGRTMAX;++num)
      {
	MUST_BE_ZERO(sigaddset(&oldset,num));
	MUST_BE_ZERO(sigaction(num,&tmpact,NULL));
      }
    
    MUST_BE_ZERO(sigaddset(&oldset,SIGPROF));
    MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&oldset,0));
  }
};

static AdjustSigs global_adjust;


// -----------------------------------------------

unsigned int* setStacktop()
{
  unsigned int* top;
  unsigned int* ebp;
  getBP(ebp);
  ebp = (unsigned int*)(*ebp);
  top = (unsigned int*)(*ebp);

  void* sta[30];
  int depth = backtrace(sta,30);
  // int cnt = depth;
  if(depth > 1) depth-=1;

  //fprintf(stderr,"depth=%d top=%8.8x\n",depth,top);
  while(depth>3)
    {
      //fprintf(stderr,"depth=%d top=%8.8x\n",depth,top);
      if(top<(unsigned int*)0x10)
      {
        fprintf(stderr,"problem\n");
      }
      top=(unsigned int*)(*top);
      --depth;
    }
  //fprintf(stderr,"depth=%d top=%8.8x\n",depth,top);
  //backtrace_symbols_fd(sta,cnt,1);

#if 0
  Dl_info look;
  for(int i=0;i<cnt;++i)
    {
      if(dladdr(sta[i],&look)!=0)
	{
	  cerr << look.dli_fname 
	       << ":" << (look.dli_saddr ? look.dli_sname : "?")
	       << ":" << look.dli_saddr << "\n";
	}
      else cerr << "NONE for " << sta[i] << "\n";
    }
#endif

  return top;
}
