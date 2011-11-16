
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309
#endif

#if __linux
# include <execinfo.h>
# include <ucontext.h>
# include <sys/syscall.h>
#endif
#if __APPLE__
#include <signal.h>
//the following is needed to use the deprecated ucontext method on OS X
#define _XOPEN_SOURCE
#endif


#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>

#include <cassert>
#include <cstdio>
#include <cstring>
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

#ifdef __linux
#ifndef __USE_POSIX199309
#error "SimpleProfile requires the definition of __USE_POSIX199309"
#endif
#endif

namespace INSTR {
  typedef unsigned char byte;
  byte const RET = 0xc3;
}

namespace {

  std::string makeFileName() {
    pid_t p = getpid();
    std::ostringstream ost;
    ost << "profdata_" << p;
    return ost.str();
  }
}

// ---------------------------------------------------------------------
// Macros, for this compilation unit only
// ---------------------------------------------------------------------

#define MUST_BE_ZERO(fun) if((fun) != 0) { perror("function failed"); abort(); }
#define getBP(X)  asm ( "movl %%ebp,%0" : "=m" (X) )
#define getSP(X)  asm ( "movl %%esp,%0" : "=m" (X) )

#if 0
#define DODEBUG if(1) std::cerr
#else
#define DODEBUG if(0) std::cerr
#endif

#ifndef __USE_GNU
int const REG_EIP = 14;
int const REG_EBP = 6;
int const REG_ESP = 7;
#endif

#include "unistd.h"
#include <sstream>
#include <fstream>


namespace {

  // Record the dynamic library mapping information from /proc/ for
  // later use in discovering the names of functions recorded as
  // 'unknown_*'.
  void write_maps() {
    pid_t pid = getpid();
    std::ostringstream strinput, stroutput;
    strinput << "/proc/" << pid << "/maps";
    stroutput << "profdata_" << pid << "_maps";
    std::ifstream input(strinput.str().c_str());
    std::ofstream output(stroutput.str().c_str());
    std::string line;
    while(getline(input, line)) output << line << '\n';
    input.close();
    output.close();
  }

  FILE* frame_cond = 0;

  void openCondFile() {
    std::string filename(makeFileName());
    filename += "_condfile";
    frame_cond = fopen(filename.c_str(), "w");
    if(frame_cond == 0) {
        std::cerr << "bad open of profdata_condfile\n";
        throw std::runtime_error("bad open");
    }
  }

  void writeCond(int code) {
    fwrite(&code, sizeof(int), 1, frame_cond);
  }

  void closeCondFile() {
    fclose(frame_cond);
  }

#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
  void dumpStack(char const*, unsigned int*, unsigned int*, unsigned char*, ucontext_t*) {
    throw std::logic_error("Cannot dumpStack on 64 bit build");
  }
#else
  void dumpStack(char const* msg,
                 unsigned int* esp, unsigned int* ebp, unsigned char* eip,
                 ucontext_t* ucp) {
    fprintf(frame_cond, msg);
    fflush(frame_cond);
    return;
    fprintf(frame_cond, "dumpStack:\n i= %p\n eip[0]= %2.2x\nb= %p\n s= %p\n b[0]= %x\n b[1]= %x\n b[2]= %x\n",
            eip, eip[0], (void*)ebp, (void*)esp, ebp[0], ebp[1], ebp[2]);
    fflush(frame_cond);

#if 0
    unsigned int* spp = esp;
    for(int i = 15; i > -5; --i) {
        fprintf(frame_cond, "    %x esp[%d]= %x\n", (void*)(spp + i), i, (void*)*(spp + i));
        fflush(frame_cond);
      }
#else
    while(ucp->uc_link) {
        fprintf(frame_cond, "   %p\n", (void*)ucp->uc_link);
        ucp = ucp->uc_link;
    }
#endif
  }
#endif
}

namespace {
  // counters
  int samples_total = 0;
  int samples_missing_framepointer = 0;
}

// ---------------------------------------------------------------------
// This is the stack trace discovery code from IgHookTrace.
// ---------------------------------------------------------------------

#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
# define HAVE_UNWIND_BACKTRACE 1
#endif
#if !defined MAP_ANONYMOUS && defined MAP_ANON
# define MAP_ANONYMOUS MAP_ANON
#endif

#if HAVE_UNWIND_BACKTRACE
struct IgHookTraceArgs { void** array; int count; int size; };
extern "C" {
  typedef unsigned _Unwind_Ptr __attribute__((__mode__(__pointer__)));
  struct _Unwind_Context;
  enum _Unwind_Reason_Code
    {
      _URC_NO_REASON = 0,
      _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
      _URC_FATAL_PHASE2_ERROR = 2,
      _URC_FATAL_PHASE1_ERROR = 3,
      _URC_NORMAL_STOP = 4,
      _URC_END_OF_STACK = 5,
      _URC_HANDLER_FOUND = 6,
      _URC_INSTALL_CONTEXT = 7,
      _URC_CONTINUE_UNWIND = 8
    };

  typedef _Unwind_Reason_Code (*_Unwind_Trace_Fn) (_Unwind_Context*, void*);
  _Unwind_Reason_Code _Unwind_Backtrace (_Unwind_Trace_Fn, void*);
  _Unwind_Ptr _Unwind_GetIP (_Unwind_Context*);
}

#ifndef __linux
static _Unwind_Reason_Code
GCCBackTrace (_Unwind_Context* context, void* arg) {
  IgHookTraceArgs* args = (IgHookTraceArgs*)arg;
  if(args->count >= 0 && args->count < args->size)
    args->array [args->count++] = (void*)_Unwind_GetIP (context);
  else
    return _URC_END_OF_STACK;
  return _URC_NO_REASON;
}
#endif
#endif

int stacktrace (void* addresses[], int nmax) {
  ++samples_total;
#if __linux && __i386
# if ! __x86_64__
#  define PROBABLY_VSYSCALL_PAGE 0xffff0000
# else
#  define PROBABLY_VSYSCALL_PAGE 0xffffffff00000000
# endif
  struct frame {
    // Normal frame.
    frame*             ebp;
    void*              eip;
    // Signal frame stuff, put in here by kernel.
    int*               signo;
    siginfo_t*         info;
    ucontext_t*        ctx;
  };
  // register frame*      ebp __asm__ ("ebp");
  // register frame*      esp __asm__ ("esp");
  // use macros to avoid compiler warning.
#define getBP(X)  asm ( "movl %%ebp,%0" : "=m" (X) )
#define getSP(X)  asm ( "movl %%esp,%0" : "=m" (X) )
  register frame* ebp = 0;
  getBP(ebp);
  register frame* esp = 0;
  getSP(esp);

  frame*              fp = ebp;
  int                 depth = 0;

  // Add fake entry to be compatible with other methods
  if(depth < nmax) addresses[depth++] = reinterpret_cast<void*>(reinterpret_cast<unsigned long>(&stacktrace));

  // Top-most frame ends with null pointer; check the rest is reasonable
  while(depth < nmax && fp >= esp) {
      // Add this stack frame.  The return address is the
      // instruction immediately after the "call".  The call
      // instruction itself is 4 or 6 bytes; we guess 4.
      addresses[depth++] = (char*)fp->eip - 4;

      // Recognise signal frames.  We use two different methods
      // depending on the linux kernel version.
      //
      // For the "old" kernels / systems we check the instructions
      // at the caller's return address.  We take it to be a signal
      // frame if we find the signal return code sequence there
      // and the thread register context structure pointer:
      //
      //    mov $__NR_rt_sigreturn, %eax
      //    int 0x80
      //
      // For the "new" kernels / systems the operating system maps
      // a "vsyscall" page at a high address, and it may contain
      // either the above code, or use of the sysenter/sysexit
      // instructions.  We cannot poke at that page so we take the
      // the high address as an indication this is a signal frame.
      // (http://www.trilithium.com/johan/2005/08/linux-gate/)
      // (http://manugarg.googlepages.com/systemcallinlinux2_6.html)
      //
      // If we don't recognise the signal frame correctly here, we
      // lose one stack frame: signal delivery is not a call so
      // when the signal handler is entered, ebp still points to
      // what it was just before the signal.
      unsigned char* insn = (unsigned char*)fp->eip;
      if(insn
          && insn[0] == 0xb8 && insn[1] == __NR_rt_sigreturn
          && insn[5] == 0xcd && insn[6] == 0x80
          && fp->ctx) {
          void* retip = (void*)fp->ctx->uc_mcontext.gregs [REG_EIP];
          if(depth < nmax) addresses[depth++] = retip;

          fp = (frame*)fp->ctx->uc_mcontext.gregs [REG_EBP];
          if(fp && (unsigned long) retip > PROBABLY_VSYSCALL_PAGE) {
              // __kernel_vsyscall stack on system call exit is
              // [0] %ebp, [1] %edx, [2] %ecx, [3] return address.
              if(depth < nmax) addresses[depth++] = ((void**)fp)[3];
              fp = fp->ebp;

              // It seems the frame _above_ __kernel_syscall (the
              // syscall implementation in libc, such as __mmap())
              // is essentially frame-pointer-less, so we should
              // find also the call above, but I don't know how
              // to determine how many arguments the system call
              // pushed on stack to call __kernel_syscall short
              // of interpreting the DWARF unwind information :-(
              // So we may lose one level of call stack here.
              ++samples_missing_framepointer;
          }
        }

      // Otherwise it's a normal frame, process through frame pointer.
      else
        fp = fp->ebp;
    }

  return depth;
#elif __linux
  return backtrace (addresses, nmax);
#elif HAVE_UNWIND_BACKTRACE
  if(nmax >= 1) {
      IgHookTraceArgs args = { addresses, 0, nmax };
      _Unwind_Backtrace (&GCCBackTrace, &args);

      if(args.count > 1 && args.array [args.count-1] == 0)
        args.count--;

      return args.count;
  }
  return 0;
#else
  return 0;
#endif
}

// ---------------------------------------------------------------------
// The signal handler. We register this to handle the SIGPROF signal.
// ---------------------------------------------------------------------

extern "C" {
  void sigFunc(int /*sig*/, siginfo_t* /*info*/, void* /*context*/) {
    SimpleProfiler* prof = SimpleProfiler::instance();
    void** arr = prof->tempStack();
    int nmax = prof->tempStackSize();
    int stackdepth = stacktrace(arr, nmax);

    assert(stackdepth <= nmax);

    // We don't want the first three entries, because they always
    // contain information about the how the signal handler was
    // called, the signal handler, and the stacktrace function.
    prof->commitFrame(arr + 3, arr + stackdepth);
  }
}

namespace {

#if USE_SIGALTSTACK
  stack_t ss_area;
#endif

  void setupTimer();

  void* setStacktop() {
#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
    throw std::logic_error("setStacktop not callable on 64 bit platform");
    return 0;
#else
    std::string consttarget_name("__libc_start_main");
    unsigned int* ebp_reg;
    getBP(ebp_reg);
    unsigned int* ebp = (unsigned int*)(ebp_reg);
    unsigned int* top = (unsigned int*)(ebp);

    void* address_of_target = 0;
    int const max_stackdepth = 30;
    void* sta[max_stackdepth];

    int depth = backtrace(sta, max_stackdepth);
    int cnt = depth;
    if(depth > 1) depth -= 1;

    // find function one below main()
    Dl_info look;
    for(int i = 0; i < cnt && !address_of_target; ++i) {
        if(dladdr(sta[i], &look) != 0) {
            if(look.dli_saddr && target_name == look.dli_sname) {
                address_of_target = sta[i];
            }
        } else {
            // This isn't really an error; it just means the function
            // was not found by the dynamic loader. The function might
            // be one that is declared 'static', and is thus not
            // visible outside of its compilation unit.
            std::cerr << "setStacktop: no function information for "
                      << sta[i]
                      << "\n";
        }
    }

    if(address_of_target == 0) {
      throw std::runtime_error("no main function found in stack");
    }

    //fprintf(stderr, "depth=%d top=%8.8x\n", depth, top);

    // Now we walk toward the beginning of the stack until we find the
    // frame that is holding the return address of our target function.

    // depth is how many more frames there are to look at in the stack.
    // top is the frame we are currently looking at.
    // top + 1 is the address to which the current frame will return.
    while(depth > 0 && (void*)*(top + 1) != address_of_target) {
        //fprintf(stderr, "depth=%d top=%8.8x func=%8.8x\n", depth, top, *(top + 1));
        if(top < (unsigned int*)0x10) fprintf(stderr, "problem\n");
        top = (unsigned int*)(*top);
        --depth;
    };

    if(depth == 0) {
      throw std::runtime_error("setStacktop: could not find stack bottom");
    }

    // Now we have to move one frame more, to the frame of the target
    // function. We want the location in memory of this frame (not any
    // address stored in the frame, but the address of the frame
    // itself).
    top = (unsigned int*)(*top);

    return top;
#endif
  }

  void setupTimer() {
#if USE_SIGALTSTACK
    static vector<char> charbuffer(SIGSTKSZ);
    //ss_area.ss_sp = new char[SIGSTKSZ];
    ss_area.ss_sp = &charbuffer[0];
    ss_area.ss_flags = 0;
    ss_area.ss_size = SIGSTKSZ;
#endif
    //static int is_set = 0;
    //if(is_set != 1) { ++is_set; return; }
    //else ++is_set;

    sigset_t myset, oldset;
    // all blocked for now
    MUST_BE_ZERO(sigfillset(&myset));
    MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK, &myset, &oldset));

#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
#if __linux
    // ignore all the RT signals
    struct sigaction tmpact;
    memset(&tmpact, 0, sizeof(tmpact));
    tmpact.sa_handler = SIG_IGN;

    for(int num=SIGRTMIN; num < SIGRTMAX; ++num) {
        MUST_BE_ZERO(sigaddset(&oldset, num));
        MUST_BE_ZERO(sigaction(num, &tmpact, NULL));
    }
#endif
#endif

#if USE_SIGALTSTACK
    if(sigaltstack(&ss_area, 0) != 0) {
        perror("sigaltstack failed for profile timer interrupt");
        abort();
    }
#endif

    // set up my RT signal now
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_sigaction = sigFunc;
    act.sa_flags = SA_RESTART | SA_SIGINFO | SA_ONSTACK;

    // get my signal number
    int mysig = SIGPROF;

    if(sigaction(mysig, &act, NULL) != 0) {
        perror("sigaction failed");
        abort();
    }

    // Turn off handling of SIGSEGV signal
    memset(&act, 0, sizeof(act));
    act.sa_handler = SIG_DFL;

    if(sigaction(SIGSEGV, &act, NULL) != 0) {
        perror("sigaction failed");
        abort();
    }

    struct rlimit limits;
    if(getrlimit(RLIMIT_CORE, &limits) != 0) {
        perror("getrlimit failed");
        abort();
    }
    std::cerr << "core size limit (soft): " << limits.rlim_cur << '\n';
    std::cerr << "core size limit (hard): " << limits.rlim_max << '\n';

    struct itimerval newval;
    struct itimerval oldval;

    newval.it_interval.tv_sec = 0;
    newval.it_interval.tv_usec = 10000;
    newval.it_value.tv_sec = 0;
    newval.it_value.tv_usec = 10000;

    if(setitimer(ITIMER_PROF, &newval, &oldval) != 0) {
        perror("setitimer failed");
        abort();
    }

    // reenable the signals, including my interval timer
    MUST_BE_ZERO(sigdelset(&oldset, mysig));
    MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK, &oldset, 0));
  }

  struct AdjustSigs {
    AdjustSigs()
    {
      sigset_t myset, oldset;
      // all blocked for now
      MUST_BE_ZERO(sigfillset(&myset));
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK, &myset, &oldset));

      // ignore all the RT signals
      struct sigaction tmpact;
      memset(&tmpact, 0, sizeof(tmpact));
      tmpact.sa_handler = SIG_IGN;

#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
#if __linux
      for(int num=SIGRTMIN; num<SIGRTMAX; ++num) {
          MUST_BE_ZERO(sigaddset(&oldset, num));
          MUST_BE_ZERO(sigaction(num, &tmpact, NULL));
      }
#endif
#endif

      MUST_BE_ZERO(sigaddset(&oldset, SIGPROF));
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK, &oldset, 0));
    }
  };

  static AdjustSigs global_adjust;
}

SimpleProfiler* SimpleProfiler::inst_ = 0;
boost::mutex SimpleProfiler::lock_;


SimpleProfiler* SimpleProfiler::instance() {
  if(SimpleProfiler::inst_ == 0) {
      boost::mutex::scoped_lock sl(lock_);
      if(SimpleProfiler::inst_ == 0) {
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
  stacktop_(setStacktop()) {
  if(fd_ < 0) {
    std::ostringstream ost;
    ost << "failed to open profiler output file " << filename_;
    throw std::runtime_error(ost.str().c_str());
  }
  openCondFile();
}

SimpleProfiler::~SimpleProfiler() {
  if(running_) {
    std::cerr << "Warning: the profile timer was not stopped,\n"
              << "profiling data in " << filename_
              << " is probably incomplete and will not be processed\n";
  }
  closeCondFile();
}

void SimpleProfiler::commitFrame(void** first, void** last) {
  void** cnt_ptr = curr_;
  *cnt_ptr = reinterpret_cast<void*>(std::distance(first, last));
  ++curr_;
  curr_ = std::copy(first, last, curr_);
  if(curr_ > high_water_) doWrite();
}

void SimpleProfiler::doWrite() {
  void** start = &frame_data_[0];
  if(curr_ == start) return;
  unsigned int cnt = std::distance(start, curr_) * sizeof(void*);
  unsigned int totwr = 0;

  while(cnt>0 && (totwr = write(fd_, start, cnt)) != cnt) {
    if(totwr == 0)
      throw std::runtime_error("SimpleProfiler::doWrite wrote zero bytes");
    start += (totwr/sizeof(void*));
    cnt -= totwr;
  }

  curr_ = &frame_data_[0];
}

#if defined(__x86_64__) || defined(__LP64__) || defined(_LP64)
void SimpleProfiler::start() {
  throw std::logic_error("SimpleProfiler not available on 64 bit platform");
}
#else
void SimpleProfiler::start() {
  {
    boost::mutex::scoped_lock sl(lock_);

    if(installed_) {
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
#endif

void SimpleProfiler::stop() {
  if(!installed_) {
      std::cerr << "SimpleProfiler::stop - no timer installed to stop\n";
      return;
  }

  if(owner_ != pthread_self()) {
      std::cerr << "SimpleProfiler::stop - only owning thread can stop the timer\n";
      return;
  }

  if(!running_) {
      std::cerr << "SimpleProfiler::stop - no timer is running\n";
      return;
  }

  struct itimerval newval;

  newval.it_interval.tv_sec = 0;
  newval.it_interval.tv_usec = 0;
  newval.it_value.tv_sec = 0;
  newval.it_value.tv_usec = 0;

  if(setitimer(ITIMER_PROF, &newval, 0) != 0) {
      perror("setitimer call failed - could not stop the timer");
  }

  running_ = false;
  complete();
}

void SimpleProfiler::complete() {
  doWrite();

  if(lseek(fd_, 0, SEEK_SET) < 0) {
      std::cerr << "SimpleProfiler: could not seek to the start of the profile\n"
                << " data file during completion.  Data will be lost.\n";
      return;
  }

  writeProfileData(fd_, filename_);
  write_maps();

  std::string totsname = makeFileName();
  totsname += "_sample_info";
  std::ofstream ost(totsname.c_str());

  ost << "samples_total " << samples_total << "\n"
      << "samples_missing_framepointer " << samples_missing_framepointer << "\n" ;
}
