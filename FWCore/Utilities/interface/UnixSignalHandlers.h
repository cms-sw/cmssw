#ifndef FWCore_Utilities_UnixSignalHandlers_h
#define FWCore_Utilities_UnixSignalHandlers_h

/*----------------------------------------------------------------------

UnixSignalHandlers: A set of little utility functions to establish
and manipulate Unix-style signal handling.

At present, only SIGUSR2 is considered.

----------------------------------------------------------------------*/

#include <string>
#include <signal.h>

#include "sigc++/signal.h"
#include "boost/thread/thread.hpp"

namespace edm {

    extern "C" {
      void ep_sigusr2(int,siginfo_t*,void*);
      extern volatile bool shutdown_flag;
      typedef void(*CFUNC)(int,siginfo_t*,void*);
    }

    int getSigNum();
    void disableAllSigs(sigset_t* oldset);
    void disableRTSigs();
    void reenableSigs(sigset_t* oldset);
    void installSig(int signum, CFUNC func);

}  // end of namespace edm
#endif
