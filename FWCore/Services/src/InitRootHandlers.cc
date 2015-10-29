#include "FWCore/Services/src/InitRootHandlers.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include <thread>
#include <sys/wait.h>
#include <sstream>
#include <string.h>
#include <poll.h>

// WORKAROUND: At CERN, execv is replaced with a non-async-signal safe
// version.  This can break our stack trace printer.  Avoid this by
// invoking the syscall directly.
#ifdef __linux__
#include <syscall.h>
#endif

#include "TROOT.h"
#include "TError.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TH1.h"
#include "TSystem.h"
#include "TUnixSystem.h"
#include "TTree.h"
#include "TVirtualStreamerInfo.h"

#include "TThread.h"
#include "TClassTable.h"

namespace edm {
  namespace service {
    int cmssw_stacktrace(void *);
  }
}

namespace {
  enum class SeverityLevel {
    kInfo,
    kWarning,
    kError,
    kSysError,
    kFatal
  };
  
  static thread_local bool s_ignoreWarnings = false;

  static bool s_ignoreEverything = false;

  void RootErrorHandlerImpl(int level, char const* location, char const* message) {

  bool die = false;

  // Translate ROOT severity level to MessageLogger severity level

    SeverityLevel el_severity = SeverityLevel::kInfo;

    if (level >= kFatal) {
      el_severity = SeverityLevel::kFatal;
    } else if (level >= kSysError) {
      el_severity = SeverityLevel::kSysError;
    } else if (level >= kError) {
      el_severity = SeverityLevel::kError;
    } else if (level >= kWarning) {
      el_severity = s_ignoreWarnings ? SeverityLevel::kInfo : SeverityLevel::kWarning;
    }

    if(s_ignoreEverything) {
      el_severity = SeverityLevel::kInfo;
    }

  // Adapt C-strings to std::strings
  // Arrange to report the error location as furnished by Root

    std::string el_location = "@SUB=?";
    if (location != 0) el_location = std::string("@SUB=")+std::string(location);

    std::string el_message  = "?";
    if (message != 0) el_message  = message;

  // Try to create a meaningful id string using knowledge of ROOT error messages
  //
  // id ==     "ROOT-ClassName" where ClassName is the affected class
  //      else "ROOT/ClassName" where ClassName is the error-declaring class
  //      else "ROOT"

    std::string el_identifier = "ROOT";

    std::string precursor("class ");
    size_t index1 = el_message.find(precursor);
    if (index1 != std::string::npos) {
      size_t index2 = index1 + precursor.length();
      size_t index3 = el_message.find_first_of(" :", index2);
      if (index3 != std::string::npos) {
        size_t substrlen = index3-index2;
        el_identifier += "-";
        el_identifier += el_message.substr(index2,substrlen);
      }
    } else {
      index1 = el_location.find("::");
      if (index1 != std::string::npos) {
        el_identifier += "/";
        el_identifier += el_location.substr(0, index1);
      }
    }

  // Intercept some messages and upgrade the severity

      if ((el_location.find("TBranchElement::Fill") != std::string::npos)
       && (el_message.find("fill branch") != std::string::npos)
       && (el_message.find("address") != std::string::npos)
       && (el_message.find("not set") != std::string::npos)) {
        el_severity = SeverityLevel::kFatal;
      }

      if ((el_message.find("Tree branches") != std::string::npos)
       && (el_message.find("different numbers of entries") != std::string::npos)) {
        el_severity = SeverityLevel::kFatal;
      }


  // Intercept some messages and downgrade the severity

      if ((el_message.find("no dictionary for class") != std::string::npos) ||
          (el_message.find("already in TClassTable") != std::string::npos) ||
          (el_message.find("matrix not positive definite") != std::string::npos) ||
          (el_message.find("not a TStreamerInfo object") != std::string::npos) ||
          (el_message.find("Problems declaring payload") != std::string::npos) ||
          (el_message.find("Announced number of args different from the real number of argument passed") != std::string::npos) || // Always printed if gDebug>0 - regardless of whether warning message is real.
          (el_location.find("Fit") != std::string::npos) ||
          (el_location.find("TDecompChol::Solve") != std::string::npos) ||
          (el_location.find("THistPainter::PaintInit") != std::string::npos) ||
          (el_location.find("TUnixSystem::SetDisplay") != std::string::npos) ||
          (el_location.find("TGClient::GetFontByName") != std::string::npos) ||
	        (el_message.find("nbins is <=0 - set to nbins = 1") != std::string::npos) ||
	        (el_message.find("nbinsy is <=0 - set to nbinsy = 1") != std::string::npos) ||
          (level < kError and
           (el_location.find("CINTTypedefBuilder::Setup")!= std::string::npos) and
           (el_message.find("possible entries are in use!") != std::string::npos))) {
        el_severity = SeverityLevel::kInfo;
      }

    if (el_severity == SeverityLevel::kInfo) {
      // Don't throw if the message is just informational.
      die = false;
    } else {
      die = true;
    }

  // Feed the message to the MessageLogger and let it choose to suppress or not.

  // Root has declared a fatal error.  Throw an EDMException unless the
  // message corresponds to a pending signal. In that case, do not throw
  // but let the OS deal with the signal in the usual way.
    if (die && (el_location != std::string("@SUB=TUnixSystem::DispatchSignals"))) {
       std::ostringstream sstr;
       sstr << "Fatal Root Error: " << el_location << "\n" << el_message << '\n';
       edm::Exception except(edm::errors::FatalRootError, sstr.str());
       except.addAdditionalInfo(except.message());
       except.clearMessage();
       throw except;
     
    }

    // Typically, we get here only for informational messages,
    // but we leave the other code in just in case we change
    // the criteria for throwing.
    if (el_severity == SeverityLevel::kFatal) {
      edm::LogError("Root_Fatal") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kSysError) {
      edm::LogError("Root_Severe") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kError) {
      edm::LogError("Root_Error") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kWarning) {
      edm::LogWarning("Root_Warning") << el_location << el_message ;
    } else if (el_severity == SeverityLevel::kInfo) {
      edm::LogInfo("Root_Information") << el_location << el_message ;
    }
  }

  void RootErrorHandler(int level, bool, char const* location, char const* message) {
    RootErrorHandlerImpl(level, location, message);
  }

  extern "C" {

    static int full_write(int fd, const char *text)
    {
      const char *buffer = text;
      size_t count = strlen(text);
      ssize_t written = 0;
      while (count)
      {
        written = write(fd, buffer, count);
        if (written == -1) 
        {
          if (errno == EINTR) {continue;}
          else {return -errno;}
        }
        count -= written;
        buffer += written;
      }
      return 0;
    }

    static int full_read(int fd, char *inbuf, size_t len, int timeout_s=-1)
    {
      char *buf = inbuf;
      size_t count = len;
      ssize_t complete = 0;
      std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
      int flags;
      if (timeout_s < 0)
      {
        flags = O_NONBLOCK;  // Prevents us from trying to set / restore flags later.
      }
      else if ((-1 == (flags = fcntl(fd, F_GETFL))))
      {
        return -errno;
      }
      if ((flags & O_NONBLOCK) != O_NONBLOCK)
      {
        if (-1 == fcntl(fd, F_SETFL, flags | O_NONBLOCK))
        {
          return -errno;
        }
      }
      while (count)
      {
        if (timeout_s >= 0)
        {
          struct pollfd poll_info{fd, POLLIN, 0};
          int ms_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-std::chrono::steady_clock::now()).count();
          if (ms_remaining > 0)
          {
            if (poll(&poll_info, 1, ms_remaining) == 0)
            {
              if ((flags & O_NONBLOCK) != O_NONBLOCK)
              {
                fcntl(fd, F_SETFL, flags);
              }
              return -ETIMEDOUT;
            }
          }
          else if (ms_remaining < 0)
          {
            if ((flags & O_NONBLOCK) != O_NONBLOCK)
            {
              fcntl(fd, F_SETFL, flags);
            }
            return -ETIMEDOUT;
          }
        }
        complete = read(fd, buf, count);
        if (complete == -1)
        {
          if (errno == EINTR) {continue;}
          else if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) {continue;}
          else
          {
            int orig_errno = errno;
            if ((flags & O_NONBLOCK) != O_NONBLOCK)
            {
              fcntl(fd, F_SETFL, flags);
            }
            return -orig_errno;
          }
        }
        count -= complete;
        buf += complete;
      }
      if ((flags & O_NONBLOCK) != O_NONBLOCK) {
        fcntl(fd, F_SETFL, flags);
      }
      return 0;
    }

    static int full_cerr_write(const char *text)
    {
      return full_write(2, text);
    }

    void sig_dostack_then_abort(int sig, siginfo_t*, void*) {

      const char* signalname = "unknown";
      switch (sig) {
        case SIGBUS:
            signalname = "bus error";
            break;
          case SIGSEGV:
            signalname = "segmentation violation";
            break;
          case SIGILL:
            signalname = "illegal instruction"; 
          default:
            break;
      }
      full_cerr_write("\n\nA fatal system signal has occurred: ");
      full_cerr_write(signalname);
      full_cerr_write("\nThe following is the call stack containing the origin of the signal.\n"
        "NOTE:The first few functions on the stack are artifacts of processing the signal and can be ignored\n\n");

      edm::service::InitRootHandlers::stacktraceFromThread();

      full_cerr_write("\nA fatal system signal has occurred: ");
      full_cerr_write(signalname);
      full_cerr_write("\n");

      // For these three known cases, re-raise the signal so get the correct
      // exit code.
      if ((sig == SIGILL) || (sig == SIGSEGV) || (sig == SIGBUS))
      {
        signal(sig, SIG_DFL);
        raise(sig);
      }
      else
      {
        ::abort();
      }
    }
    
    void sig_abort(int sig, siginfo_t*, void*) {
      ::abort();
    }
  }

  void set_default_signals() {
    signal(SIGILL, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
  }

}  // end of unnamed namespace

namespace edm {
  namespace service {

    /*
     * We've run into issues where GDB fails to print the thread which calls clone().
     * To avoid this problem, we have an alternate approach below where the signal handler
     * only reads/writes to a dedicated thread via pipes.  The helper thread does the clone()
     * invocation; we don't care if that thread is missing from the traceback in this case.
     */
    static void cmssw_stacktrace_fork();

    void InitRootHandlers::stacktraceHelperThread()
    {
      int toParent = childToParent_[1];
      int fromParent = parentToChild_[0];
      char buf[2]; buf[1] = '\0';
      while(true)
      {
        int result = full_read(fromParent, buf, 1);
        if (result < 0)
        {
          // To avoid a deadlock (this function is NOT re-entrant), reset signals
          // We never set them back to the CMSSW handler because we assume the parent
          // thread will abort for us.
          set_default_signals();
          close(toParent);
          full_cerr_write("\n\nTraceback helper thread failed to read from parent: ");
          full_cerr_write(strerror(-result));
          full_cerr_write("\n");
          ::abort();
        }
        if (buf[0] == '1')
        {
          set_default_signals();
          cmssw_stacktrace_fork();
          full_write(toParent, buf);
        }
        else if (buf[0] == '2')
        {
          // We have just finished forking.  Reload the file descriptors for thread
          // communication.
          close(toParent);
          close(fromParent);
          toParent = childToParent_[1];
          fromParent = parentToChild_[0];
        }
        else if (buf[0] == '3')
        {
          break;
        }
        else
        {
          set_default_signals();
          close(toParent);
          full_cerr_write("\n\nTraceback helper thread got unknown command from parent: ");
          full_cerr_write(buf);
          full_cerr_write("\n");
          ::abort();
        }
      }
    }

    void InitRootHandlers::stacktraceFromThread()
    {
      int result = full_write(parentToChild_[1], "1");
      if (result < 0)
      {
        full_cerr_write("\n\nAttempt to request stacktrace failed: ");
        full_cerr_write(strerror(-result));
        full_cerr_write("\n");
        return;
      }
      char buf[2]; buf[1] = '\0';
      if ((result = full_read(childToParent_[0], buf, 1, 5*60)) < 0)
      {
        full_cerr_write("\n\nWaiting for stacktrace completion failed: ");
        if (result == -ETIMEDOUT)
        {
          full_cerr_write("timed out waiting for GDB to complete.");
        }
        else
        {
          full_cerr_write(strerror(-result));
        }
        full_cerr_write("\n");
        return;
      }
    }

    void cmssw_stacktrace_fork()
    {
      char child_stack[4*1024];
      char *child_stack_ptr = child_stack + 4*1024;
        // On Linux, we currently use jemalloc.  This registers pthread_atfork handlers; these
        // handlers are *not* async-signal safe.  Hence, a deadlock is possible if we invoke
        // fork() from our signal handlers.  Accordingly, we use clone (not POSIX, but AS-safe)
        // as that is closer to the 'raw metal' syscall and avoids pthread_atfork handlers.
      int pid =
#ifdef __linux__
        clone(edm::service::cmssw_stacktrace, child_stack_ptr, CLONE_VM|CLONE_FS|SIGCHLD, nullptr);
#else
        fork();
      if (child_stack_ptr) {} // Suppress 'unused variable' warning on non-Linux
      if (pid == 0) {edm::service::cmssw_stacktrace(nullptr); ::abort();}
#endif
      if (pid == -1)
      {
        full_cerr_write("(Attempt to perform stack dump failed.)\n");
      }
      else
      {
        int status;
        if (waitpid(pid, &status, 0) == -1)
        {
          full_cerr_write("(Failed to wait on stack dump output.)\n");
        }
        if (status)
        {
          full_cerr_write("(GDB stack trace failed unexpectedly)\n");
        }
      }
    }

    int cmssw_stacktrace(void * /*arg*/)
    {
      char *const *argv = edm::service::InitRootHandlers::getPstackArgv();
      // NOTE: this is NOT async-signal-safe at CERN's lxplus service.
      // CERN uses LD_PRELOAD to replace execv with a function from libsnoopy which
      // calls dlsym.
#ifdef __linux__
      syscall(SYS_execve, "/bin/sh", argv, __environ);
#else
      execv("/bin/sh", argv);
#endif
      ::abort();
      return 1;
    }

    static char pstackName[] = "(CMSSW stack trace helper)";
    static char dashC[] = "-c";
    char InitRootHandlers::pidString_[InitRootHandlers::pidStringLength_] = {};
    char * const InitRootHandlers::pstackArgv_[] = {pstackName, dashC, InitRootHandlers::pidString_, nullptr};
    int InitRootHandlers::parentToChild_[2] = {-1, -1};
    int InitRootHandlers::childToParent_[2] = {-1, -1};
    std::unique_ptr<std::thread> InitRootHandlers::helperThread_;

    InitRootHandlers::InitRootHandlers (ParameterSet const& pset, ActivityRegistry& iReg)
      : RootHandlers(),
        unloadSigHandler_(pset.getUntrackedParameter<bool> ("UnloadRootSigHandler")),
        resetErrHandler_(pset.getUntrackedParameter<bool> ("ResetRootErrHandler")),
        loadAllDictionaries_(pset.getUntrackedParameter<bool>("LoadAllDictionaries")),
        autoLibraryLoader_(loadAllDictionaries_ or pset.getUntrackedParameter<bool> ("AutoLibraryLoader"))
    {
      
      if(unloadSigHandler_) {
      // Deactivate all the Root signal handlers and restore the system defaults
        gSystem->ResetSignal(kSigChild);
        gSystem->ResetSignal(kSigBus);
        gSystem->ResetSignal(kSigSegmentationViolation);
        gSystem->ResetSignal(kSigIllegalInstruction);
        gSystem->ResetSignal(kSigSystem);
        gSystem->ResetSignal(kSigPipe);
        gSystem->ResetSignal(kSigAlarm);
        gSystem->ResetSignal(kSigUrgent);
        gSystem->ResetSignal(kSigFloatingException);
        gSystem->ResetSignal(kSigWindowChanged);
      } else if(pset.getUntrackedParameter<bool>("AbortOnSignal")){
        cachePidInfo();

        //NOTE: ROOT can also be told to abort on these kinds of problems BUT
        // it requires an TApplication to be instantiated which causes problems
        gSystem->ResetSignal(kSigBus);
        gSystem->ResetSignal(kSigSegmentationViolation);
        gSystem->ResetSignal(kSigIllegalInstruction);
        installCustomHandler(SIGBUS,sig_dostack_then_abort);
        sigBusHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGBUS,sig_abort);
        });
        installCustomHandler(SIGSEGV,sig_dostack_then_abort);
        sigSegvHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGSEGV,sig_abort);
        });
        installCustomHandler(SIGILL,sig_dostack_then_abort);
        sigIllHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGILL,sig_abort);
        });
        iReg.watchPostForkReacquireResources(this, &InitRootHandlers::cachePidInfoHandler);
      }

      if(resetErrHandler_) {

      // Replace the Root error handler with one that uses the MessageLogger
        SetErrorHandler(RootErrorHandler);
      }

      // Enable automatic Root library loading.
      if(autoLibraryLoader_) {
        gInterpreter->SetClassAutoloading(1);
      }

      // Set ROOT parameters.
      TTree::SetMaxTreeSize(kMaxLong64);
      TH1::AddDirectory(kFALSE);
      //G__SetCatchException(0);

      // Set custom streamers
      setRefCoreStreamer();

      // Load the library containing dictionaries for std:: classes, if not already loaded.
      if (!hasDictionary(typeid(std::vector<std::vector<unsigned int> >))) {
         TypeWithDict::byName("std::vector<std::vector<unsigned int> >");
      }

      int debugLevel = pset.getUntrackedParameter<int>("DebugLevel");
      if(debugLevel >0) {
	gDebug = debugLevel;
      }
    }

    InitRootHandlers::~InitRootHandlers () {
      // close all open ROOT files
      TIter iter(gROOT->GetListOfFiles());
      TObject *obj = nullptr;
      while(nullptr != (obj = iter.Next())) {
        TFile* f = dynamic_cast<TFile*>(obj);
        if(f) {
          // We get a new iterator each time,
          // because closing a file can invalidate the iterator
          f->Close();
          iter = TIter(gROOT->GetListOfFiles());
        }
      }
    }
    
    void InitRootHandlers::willBeUsingThreads() {
      //Tell Root we want to be multi-threaded
      TThread::Initialize();
      //When threading, also have to keep ROOT from logging all TObjects into a list
      TObject::SetObjectStat(false);
      
      //Have to avoid having Streamers modify themselves after they have been used
      TVirtualStreamerInfo::Optimize(false);
    }
    
    void InitRootHandlers::initializeThisThreadForUse() {
      static thread_local TThread guard;
    }

    void InitRootHandlers::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Centralized interface to ROOT.");
      desc.addUntracked<bool>("UnloadRootSigHandler", false)
          ->setComment("If True, signals are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("ResetRootErrHandler", true)
          ->setComment("If True, ROOT messages (e.g. errors, warnings) are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("AutoLibraryLoader", true)
          ->setComment("If True, enables automatic loading of data dictionaries.");
      desc.addUntracked<bool>("LoadAllDictionaries",false)
          ->setComment("If True, loads all ROOT dictionaries.");
      desc.addUntracked<bool>("AbortOnSignal",true)
      ->setComment("If True, do an abort when a signal occurs that causes a crash. If False, ROOT will do an exit which attempts to do a clean shutdown.");
      desc.addUntracked<int>("DebugLevel",0)
 	  ->setComment("Sets ROOT's gDebug value.");
      descriptions.add("InitRootHandlers", desc);
    }

    char *const *
    InitRootHandlers::getPstackArgv() {
      return pstackArgv_;
    }

    void
    InitRootHandlers::enableWarnings_() {
      s_ignoreWarnings =false;
    }

    void
    InitRootHandlers::ignoreWarnings_() {
      s_ignoreWarnings = true;
    }

    void
    InitRootHandlers::cachePidInfo()
    {
      if (snprintf(pidString_, pidStringLength_-1, "gdb -quiet -p %d 2>&1 <<EOF |\n"
        "set width 0\n"
        "set height 0\n"
        "set pagination no\n"
        "thread apply all bt\n"
        "EOF\n"
        "/bin/sed -n -e 's/^\\((gdb) \\)*//' -e '/^#/p' -e '/^Thread/p'", getpid()) >= pidStringLength_)
      {
        std::ostringstream sstr;
        sstr << "Unable to pre-allocate stacktrace handler information";
        edm::Exception except(edm::errors::OtherCMS, sstr.str());
        throw except;
      }

      // These are initialized to -1; harmless to close an invalid FD.
      // If this is called post-fork, we don't want to be communicating on
      // these FDs as they are used internally by the parent.
      close(childToParent_[0]);
      close(childToParent_[1]);
      childToParent_[0] = -1; childToParent_[1] = -1;
      close(parentToChild_[0]);
      close(parentToChild_[1]);
      parentToChild_[0] = -1; parentToChild_[1] = -1;

      if (-1 == pipe2(childToParent_, O_CLOEXEC))
      {
        std::ostringstream sstr;
        sstr << "Failed to create child-to-parent pipes (errno=" << errno << "): " << strerror(errno);
        edm::Exception except(edm::errors::OtherCMS, sstr.str());
        throw except;
      }

      if (-1 == pipe2(parentToChild_, O_CLOEXEC))
      {
        close(childToParent_[0]); close(childToParent_[1]);
        childToParent_[0] = -1; childToParent_[1] = -1;
        std::ostringstream sstr;
        sstr << "Failed to create child-to-parent pipes (errno=" << errno << "): " << strerror(errno);
        edm::Exception except(edm::errors::OtherCMS, sstr.str());
        throw except;
      }

      helperThread_.reset(new std::thread(stacktraceHelperThread));
      helperThread_->detach();
    }

  }  // end of namespace service
}  // end of namespace edm
