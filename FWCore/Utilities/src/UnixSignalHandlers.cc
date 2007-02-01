#include <iostream>
#include <cstdlib>
#include <signal.h>

#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "boost/thread/thread.hpp"

using namespace std;

namespace edm {

    extern "C" {
      volatile bool shutdown_flag = false;

      void ep_sigusr2(int,siginfo_t*,void*)
      {
	FDEBUG(1) << "in sigusr2 handler\n";
	shutdown_flag = true;
      }
    }

    boost::mutex signum_lock;
    volatile int signum_value = 
#if defined(__linux__)
      SIGRTMIN;
#else
    0;
#endif

    int getSigNum()
    {
      boost::mutex::scoped_lock sl(signum_lock);
      int rc = signum_value;
      ++signum_value;
      return rc;
    }

#define MUST_BE_ZERO(fun) if((fun) != 0)					\
      { perror("UnixSignalService::setupSignal: sig function failed"); abort(); }

    void disableAllSigs(sigset_t* oldset)
    {
      sigset_t myset;
      // all blocked for now
      MUST_BE_ZERO(sigfillset(&myset));
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,&myset,oldset));
    }

    void disableRTSigs()
    {
#if defined(__linux__)
      // ignore all the RT signals
      sigset_t myset;
      MUST_BE_ZERO(sigemptyset(&myset));
      
      struct sigaction tmpact;
      memset(&tmpact,0,sizeof(tmpact));
      tmpact.sa_handler = SIG_IGN;

      for(int num = SIGRTMIN; num < SIGRTMAX; ++num) {
	  MUST_BE_ZERO(sigaddset(&myset,num));
	  MUST_BE_ZERO(sigaction(num,&tmpact,NULL));
      }
      
      MUST_BE_ZERO(pthread_sigmask(SIG_BLOCK,&myset,0));
#endif
    }

    void reenableSigs(sigset_t* oldset)
    {
      // reenable the signals
      MUST_BE_ZERO(pthread_sigmask(SIG_SETMASK,oldset,0));
    }

    void installSig(int signum, CFUNC func)
    {
      // set up my RT signal now
      struct sigaction act;
      memset(&act,0,sizeof(act));
      act.sa_sigaction = func;
      act.sa_flags = SA_RESTART;
      
      // get my signal number
      int mysig = signum;
      
      if(sigaction(mysig,&act,NULL) != 0) {
	  perror("sigaction failed");
	  abort();
      }
      
      sigset_t newset;
      MUST_BE_ZERO(sigemptyset(&newset));
      MUST_BE_ZERO(sigaddset(&newset,mysig));
      MUST_BE_ZERO(pthread_sigmask(SIG_UNBLOCK,&newset,0));
    }

} // end of namespace edm
