#include "FWCore/Utilities/interface/RootHandlers.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include "oneapi/tbb/concurrent_unordered_set.h"
#include "oneapi/tbb/task.h"
#include "oneapi/tbb/task_scheduler_observer.h"
#include "oneapi/tbb/global_control.h"
#include <memory>

#include <thread>
#include <sys/wait.h>
#include <sstream>
#include <cstring>
#include <poll.h>
#include <atomic>
#include <algorithm>
#include <vector>
#include <string>
#include <array>

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

#include "TClassTable.h"

#include <memory>

namespace {
  // size of static buffer allocated for listing module names following a
  // stacktrace abort
  constexpr std::size_t moduleBufferSize = 128;
}  // namespace

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
  class ActivityRegistry;

  namespace service {
    class InitRootHandlers : public RootHandlers {
      friend int cmssw_stacktrace(void*);

    public:
      class ThreadTracker : public oneapi::tbb::task_scheduler_observer {
      public:
        typedef oneapi::tbb::concurrent_unordered_set<pthread_t> Container_type;

        ThreadTracker() : oneapi::tbb::task_scheduler_observer() { observe(); }
        ~ThreadTracker() override = default;

        void on_scheduler_entry(bool) override {
          // ensure thread local has been allocated; not necessary on Linux with
          // the current cmsRun linkage, but could be an issue if the platform
          // or linkage leads to "lazy" allocation of the thread local.  By
          // referencing it here we make sure it has been allocated and can be
          // accessed safely from our signal handler.
          edm::CurrentModuleOnThread::getCurrentModuleOnThread();
          threadIDs_.insert(pthread_self());
        }
        void on_scheduler_exit(bool) override {}
        const Container_type& IDs() { return threadIDs_; }

      private:
        Container_type threadIDs_;
      };

      explicit InitRootHandlers(ParameterSet const& pset, ActivityRegistry& iReg);
      ~InitRootHandlers() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void stacktraceFromThread();
      static const ThreadTracker::Container_type& threadIDs() {
        static const ThreadTracker::Container_type empty;
        if (threadTracker_) {
          return threadTracker_->IDs();
        }
        return empty;
      }
      static int stackTracePause() { return stackTracePause_; }

      static std::vector<std::array<char, moduleBufferSize>> moduleListBuffers_;
      static std::atomic<std::size_t> nextModule_, doneModules_;

    private:
      static char const* const* getPstackArgv();
      void enableWarnings_() override;
      void ignoreWarnings_(edm::RootHandlers::SeverityLevel level) override;
      void willBeUsingThreads() override;

      void cachePidInfo();
      static void stacktraceHelperThread();

      static constexpr int pidStringLength_ = 200;
      static char pidString_[pidStringLength_];
      static char const* const pstackArgv_[];
      static int parentToChild_[2];
      static int childToParent_[2];
      static std::unique_ptr<std::thread> helperThread_;
      static std::unique_ptr<ThreadTracker> threadTracker_;
      static int stackTracePause_;

      bool unloadSigHandler_;
      bool resetErrHandler_;
      bool loadAllDictionaries_;
      bool autoLibraryLoader_;
      bool interactiveDebug_;
      std::shared_ptr<const void> sigBusHandler_;
      std::shared_ptr<const void> sigSegvHandler_;
      std::shared_ptr<const void> sigIllHandler_;
      std::shared_ptr<const void> sigTermHandler_;
      std::shared_ptr<const void> sigAbrtHandler_;
    };

    inline bool isProcessWideService(InitRootHandlers const*) { return true; }

  }  // end of namespace service
}  // end of namespace edm

namespace edm {
  namespace service {
    int cmssw_stacktrace(void*);
  }
}  // namespace edm

namespace {
  thread_local edm::RootHandlers::SeverityLevel s_ignoreWarnings = edm::RootHandlers::SeverityLevel::kInfo;

  constexpr bool s_ignoreEverything = false;

  template <std::size_t SIZE>
  bool find_if_string(const std::string& search, const std::array<const char* const, SIZE>& substrs) {
    return (std::find_if(substrs.begin(), substrs.end(), [&search](const char* const s) -> bool {
              return (search.find(s) != std::string::npos);
            }) != substrs.end());
  }

  //Contents of a message which should be reported as an INFO not a ERROR
  constexpr std::array<const char* const, 9> in_message{
      {"no dictionary for class",
       "already in TClassTable",
       "matrix not positive definite",
       "not a TStreamerInfo object",
       "Problems declaring payload",
       "Announced number of args different from the real number of argument passed",  // Always printed if gDebug>0 - regardless of whether warning message is real.
       "nbins is <=0 - set to nbins = 1",
       "nbinsy is <=0 - set to nbinsy = 1",
       "oneapi::tbb::global_control is limiting"}};

  //Location generating messages which should be reported as an INFO not a ERROR
  constexpr std::array<const char* const, 7> in_location{{"Fit",
                                                          "TDecompChol::Solve",
                                                          "THistPainter::PaintInit",
                                                          "TUnixSystem::SetDisplay",
                                                          "TGClient::GetFontByName",
                                                          "Inverter::Dinv",
                                                          "RTaskArenaWrapper"}};

  constexpr std::array<const char* const, 3> in_message_print_error{{"number of iterations was insufficient",
                                                                     "bad integrand behavior",
                                                                     "integral is divergent, or slowly convergent"}};

  void RootErrorHandlerImpl(int level, char const* location, char const* message) {
    bool die = false;

    // Translate ROOT severity level to MessageLogger severity level

    edm::RootHandlers::SeverityLevel el_severity = edm::RootHandlers::SeverityLevel::kInfo;

    if (level >= kFatal) {
      el_severity = edm::RootHandlers::SeverityLevel::kFatal;
    } else if (level >= kSysError) {
      el_severity = edm::RootHandlers::SeverityLevel::kSysError;
    } else if (level >= kError) {
      el_severity = edm::RootHandlers::SeverityLevel::kError;
    } else if (level >= kWarning) {
      el_severity = edm::RootHandlers::SeverityLevel::kWarning;
    }

    if (s_ignoreEverything || el_severity <= s_ignoreWarnings) {
      el_severity = edm::RootHandlers::SeverityLevel::kInfo;
    }

    // Adapt C-strings to std::strings
    // Arrange to report the error location as furnished by Root

    std::string el_location = "@SUB=?";
    if (location != nullptr)
      el_location = std::string("@SUB=") + std::string(location);

    std::string el_message = "?";
    if (message != nullptr)
      el_message = message;

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
        size_t substrlen = index3 - index2;
        el_identifier += "-";
        el_identifier += el_message.substr(index2, substrlen);
      }
    } else {
      index1 = el_location.find("::");
      if (index1 != std::string::npos) {
        el_identifier += "/";
        el_identifier += el_location.substr(0, index1);
      }
    }

    // Intercept some messages and upgrade the severity

    if ((el_location.find("TBranchElement::Fill") != std::string::npos) &&
        (el_message.find("fill branch") != std::string::npos) && (el_message.find("address") != std::string::npos) &&
        (el_message.find("not set") != std::string::npos)) {
      el_severity = edm::RootHandlers::SeverityLevel::kFatal;
    }

    if ((el_message.find("Tree branches") != std::string::npos) &&
        (el_message.find("different numbers of entries") != std::string::npos)) {
      el_severity = edm::RootHandlers::SeverityLevel::kFatal;
    }

    // Intercept some messages and downgrade the severity

    if (find_if_string(el_message, in_message) || find_if_string(el_location, in_location) ||
        (level < kError and (el_location.find("CINTTypedefBuilder::Setup") != std::string::npos) and
         (el_message.find("possible entries are in use!") != std::string::npos))) {
      el_severity = edm::RootHandlers::SeverityLevel::kInfo;
    }

    // These are a special case because we do not want them to
    // be fatal, but we do want an error to print.
    bool alreadyPrinted = false;
    if (find_if_string(el_message, in_message_print_error)) {
      el_severity = edm::RootHandlers::SeverityLevel::kInfo;
      edm::LogError("Root_Error") << el_location << el_message;
      alreadyPrinted = true;
    }

    if (el_severity == edm::RootHandlers::SeverityLevel::kInfo) {
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
    if (!alreadyPrinted) {
      if (el_severity == edm::RootHandlers::SeverityLevel::kFatal) {
        edm::LogError("Root_Fatal") << el_location << el_message;
      } else if (el_severity == edm::RootHandlers::SeverityLevel::kSysError) {
        edm::LogError("Root_Severe") << el_location << el_message;
      } else if (el_severity == edm::RootHandlers::SeverityLevel::kError) {
        edm::LogError("Root_Error") << el_location << el_message;
      } else if (el_severity == edm::RootHandlers::SeverityLevel::kWarning) {
        edm::LogWarning("Root_Warning") << el_location << el_message;
      } else if (el_severity == edm::RootHandlers::SeverityLevel::kInfo) {
        edm::LogInfo("Root_Information") << el_location << el_message;
      }
    }
  }

  void RootErrorHandler(int level, bool, char const* location, char const* message) {
    RootErrorHandlerImpl(level, location, message);
  }

  extern "C" {
  void set_default_signals() {
    signal(SIGILL, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
  }

  static int full_write(int fd, const char* text) {
    const char* buffer = text;
    size_t count = strlen(text);
    ssize_t written = 0;
    while (count) {
      written = write(fd, buffer, count);
      if (written == -1) {
        if (errno == EINTR) {
          continue;
        } else {
          return -errno;
        }
      }
      count -= written;
      buffer += written;
    }
    return 0;
  }

  static int full_read(int fd, char* inbuf, size_t len, int timeout_s = -1) {
    char* buf = inbuf;
    size_t count = len;
    ssize_t complete = 0;
    std::chrono::time_point<std::chrono::steady_clock> end_time =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    int flags;
    if (timeout_s < 0) {
      flags = O_NONBLOCK;  // Prevents us from trying to set / restore flags later.
    } else if ((-1 == (flags = fcntl(fd, F_GETFL)))) {
      return -errno;
    }
    if ((flags & O_NONBLOCK) != O_NONBLOCK) {
      if (-1 == fcntl(fd, F_SETFL, flags | O_NONBLOCK)) {
        return -errno;
      }
    }
    while (count) {
      if (timeout_s >= 0) {
        struct pollfd poll_info {
          fd, POLLIN, 0
        };
        int ms_remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - std::chrono::steady_clock::now()).count();
        if (ms_remaining > 0) {
          int rc = poll(&poll_info, 1, ms_remaining);
          if (rc <= 0) {
            if (rc < 0) {
              if (errno == EINTR || errno == EAGAIN) {
                continue;
              }
              rc = -errno;
            } else {
              rc = -ETIMEDOUT;
            }
            if ((flags & O_NONBLOCK) != O_NONBLOCK) {
              fcntl(fd, F_SETFL, flags);
            }
            return rc;
          }
        } else if (ms_remaining < 0) {
          if ((flags & O_NONBLOCK) != O_NONBLOCK) {
            fcntl(fd, F_SETFL, flags);
          }
          return -ETIMEDOUT;
        }
      }
      complete = read(fd, buf, count);
      if (complete == -1) {
        if (errno == EINTR) {
          continue;
        } else if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) {
          continue;
        } else {
          int orig_errno = errno;
          if ((flags & O_NONBLOCK) != O_NONBLOCK) {
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

  static int full_cerr_write(const char* text) { return full_write(2, text); }

// these signals are only used inside the stacktrace signal handler,
// so common signals can be used.  They do have to be different, since
// we do not set SA_NODEFER, and RESUME must be a signal that will
// cause sleep() to return early.
#if defined(SIGRTMAX)
#define PAUSE_SIGNAL SIGRTMAX
#define RESUME_SIGNAL SIGRTMAX - 1
#elif defined(SIGINFO)  // macOS/BSD
#define PAUSE_SIGNAL SIGINFO
#define RESUME_SIGNAL SIGALRM
#endif

  // does nothing, here only to interrupt the sleep() in the pause handler
  void sig_resume_handler(int sig, siginfo_t*, void*) {}

  // pause a thread so that a (slow) stacktrace will capture the current state
  void sig_pause_for_stacktrace(int sig, siginfo_t*, void*) {
    using namespace edm::service;

#ifdef RESUME_SIGNAL
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, RESUME_SIGNAL);
    pthread_sigmask(SIG_UNBLOCK, &sigset, nullptr);
#endif
    // sleep interrrupts on a handled delivery of the resume signal
    sleep(InitRootHandlers::stackTracePause());

    if (InitRootHandlers::doneModules_.is_lock_free() && InitRootHandlers::nextModule_.is_lock_free()) {
      auto i = InitRootHandlers::nextModule_++;
      if (i < InitRootHandlers::moduleListBuffers_.size()) {
        char* buff = InitRootHandlers::moduleListBuffers_[i].data();

        strlcpy(buff, "\nModule: ", moduleBufferSize);
        if (edm::CurrentModuleOnThread::getCurrentModuleOnThread() != nullptr) {
          strlcat(buff,
                  edm::CurrentModuleOnThread::getCurrentModuleOnThread()->moduleDescription()->moduleName().c_str(),
                  moduleBufferSize);
          strlcat(buff, ":", moduleBufferSize);
          strlcat(buff,
                  edm::CurrentModuleOnThread::getCurrentModuleOnThread()->moduleDescription()->moduleLabel().c_str(),
                  moduleBufferSize);
        } else {
          strlcat(buff, "none", moduleBufferSize);
        }
        ++edm::service::InitRootHandlers::doneModules_;
      }
    }
  }

  void sig_dostack_then_abort(int sig, siginfo_t*, void*) {
    using namespace edm::service;

    const auto& tids = InitRootHandlers::threadIDs();

    const auto self = pthread_self();
#ifdef PAUSE_SIGNAL
    if (InitRootHandlers::stackTracePause() > 0 && tids.size() > 1) {
      // install the "pause" handler
      struct sigaction act;
      act.sa_sigaction = sig_pause_for_stacktrace;
      act.sa_flags = 0;
      sigemptyset(&act.sa_mask);
      sigaction(PAUSE_SIGNAL, &act, nullptr);

      // unblock pause signal globally, resume is unblocked in the pause handler
      sigset_t pausesigset;
      sigemptyset(&pausesigset);
      sigaddset(&pausesigset, PAUSE_SIGNAL);
      sigprocmask(SIG_UNBLOCK, &pausesigset, nullptr);

      // send a pause signal to all CMSSW/TBB threads other than self
      for (auto id : tids) {
        if (self != id) {
          pthread_kill(id, PAUSE_SIGNAL);
        }
      }

#ifdef RESUME_SIGNAL
      // install the "resume" handler
      act.sa_sigaction = sig_resume_handler;
      sigaction(RESUME_SIGNAL, &act, nullptr);
#endif
    }
#endif

    const char* signalname = "unknown";
    switch (sig) {
      case SIGBUS: {
        signalname = "bus error";
        break;
      }
      case SIGSEGV: {
        signalname = "segmentation violation";
        break;
      }
      case SIGILL: {
        signalname = "illegal instruction";
        break;
      }
      case SIGTERM: {
        signalname = "external termination request";
        break;
      }
      case SIGABRT: {
        signalname = "abort signal";
        break;
      }
      default:
        break;
    }
    full_cerr_write("\n\nA fatal system signal has occurred: ");
    full_cerr_write(signalname);
    full_cerr_write("\nThe following is the call stack containing the origin of the signal.\n\n");

    edm::service::InitRootHandlers::stacktraceFromThread();

    // resume the signal handlers to store the current module; we are not guaranteed they
    // will have time to store their modules, so there is a race condition; this could be
    // avoided by storing the module information before sleeping, a change that may be
    // made when we're convinced accessing the thread-local current module is safe.
#ifdef RESUME_SIGNAL
    std::size_t notified = 0;
    if (InitRootHandlers::stackTracePause() > 0 && tids.size() > 1) {
      for (auto id : tids) {
        if (self != id) {
          if (pthread_kill(id, RESUME_SIGNAL) == 0)
            ++notified;
        }
      }
    }
#endif

    full_cerr_write("\nCurrent Modules:\n");

    // Checking tids.count(self) ensures that we only try to access the current module in
    // CMSSW/TBB threads.  Those threads access the thread-local current module at the same
    // time the thread is registered, so any lazy allocation will have been done at that
    // point.  Not necessary on Linux with the current cmsRun linkage, as the thread-local
    // is allocated at exec time, not lazily.
    if (tids.count(self) > 0) {
      char buff[moduleBufferSize] = "\nModule: ";
      if (edm::CurrentModuleOnThread::getCurrentModuleOnThread() != nullptr) {
        strlcat(buff,
                edm::CurrentModuleOnThread::getCurrentModuleOnThread()->moduleDescription()->moduleName().c_str(),
                moduleBufferSize);
        strlcat(buff, ":", moduleBufferSize);
        strlcat(buff,
                edm::CurrentModuleOnThread::getCurrentModuleOnThread()->moduleDescription()->moduleLabel().c_str(),
                moduleBufferSize);
      } else {
        strlcat(buff, "none", moduleBufferSize);
      }
      strlcat(buff, " (crashed)", moduleBufferSize);
      full_cerr_write(buff);
    } else {
      full_cerr_write("\nModule: non-CMSSW (crashed)");
    }

#ifdef PAUSE_SIGNAL
    // wait a short interval for the paused threads to resume and fill in their module
    // information, then print
    if (InitRootHandlers::doneModules_.is_lock_free()) {
      int spincount = 0;
      timespec t = {0, 1000};
      while (++spincount < 1000 && InitRootHandlers::doneModules_ < notified) {
        nanosleep(&t, nullptr);
      }
      for (std::size_t i = 0; i < InitRootHandlers::doneModules_; ++i) {
        full_cerr_write(InitRootHandlers::moduleListBuffers_[i].data());
      }
    }
#endif

    full_cerr_write("\n\nA fatal system signal has occurred: ");
    full_cerr_write(signalname);
    full_cerr_write("\n");

    // For these five known cases, re-raise the signal to get the correct
    // exit code.
    if ((sig == SIGILL) || (sig == SIGSEGV) || (sig == SIGBUS) || (sig == SIGTERM) || (sig == SIGABRT)) {
      signal(sig, SIG_DFL);
      raise(sig);
    } else {
      set_default_signals();
      ::abort();
    }
  }

  void sig_abort(int sig, siginfo_t*, void*) {
    full_cerr_write("\n\nFatal system signal has occurred during exit\n");

    // re-raise the signal to get the correct exit code
    signal(sig, SIG_DFL);
    raise(sig);

    // shouldn't get here
    set_default_signals();
    ::sleep(10);
    ::abort();
  }
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

    void InitRootHandlers::stacktraceHelperThread() {
      int toParent = childToParent_[1];
      int fromParent = parentToChild_[0];
      char buf[2];
      buf[1] = '\0';

      while (true) {
        int result = full_read(fromParent, buf, 1);
        if (result < 0) {
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
        if (buf[0] == '1') {
          set_default_signals();
          cmssw_stacktrace_fork();
          full_write(toParent, buf);
        } else if (buf[0] == '2') {
          // We have just finished forking.  Reload the file descriptors for thread
          // communication.
          close(toParent);
          close(fromParent);
          toParent = childToParent_[1];
          fromParent = parentToChild_[0];
        } else if (buf[0] == '3') {
          break;
        } else {
          set_default_signals();
          close(toParent);
          full_cerr_write("\n\nTraceback helper thread got unknown command from parent: ");
          full_cerr_write(buf);
          full_cerr_write("\n");
          ::abort();
        }
      }
    }

    void InitRootHandlers::stacktraceFromThread() {
      int result = full_write(parentToChild_[1], "1");
      if (result < 0) {
        full_cerr_write("\n\nAttempt to request stacktrace failed: ");
        full_cerr_write(strerror(-result));
        full_cerr_write("\n");
        return;
      }
      char buf[2];
      buf[1] = '\0';
      if ((result = full_read(childToParent_[0], buf, 1, 5 * 60)) < 0) {
        full_cerr_write("\n\nWaiting for stacktrace completion failed: ");
        if (result == -ETIMEDOUT) {
          full_cerr_write("timed out waiting for GDB to complete.");
        } else {
          full_cerr_write(strerror(-result));
        }
        full_cerr_write("\n");
        return;
      }
    }

    void cmssw_stacktrace_fork() {
      char child_stack[4 * 1024];
      char* child_stack_ptr = child_stack + 4 * 1024;
      // On Linux, we currently use jemalloc.  This registers pthread_atfork handlers; these
      // handlers are *not* async-signal safe.  Hence, a deadlock is possible if we invoke
      // fork() from our signal handlers.  Accordingly, we use clone (not POSIX, but AS-safe)
      // as that is closer to the 'raw metal' syscall and avoids pthread_atfork handlers.
      int pid =
#ifdef __linux__
          clone(edm::service::cmssw_stacktrace, child_stack_ptr, CLONE_VM | CLONE_FS | SIGCHLD, nullptr);
#else
          fork();
      if (child_stack_ptr) {
      }  // Suppress 'unused variable' warning on non-Linux
      if (pid == 0) {
        edm::service::cmssw_stacktrace(nullptr);
      }
#endif
      if (pid == -1) {
        full_cerr_write("(Attempt to perform stack dump failed.)\n");
      } else {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
          full_cerr_write("(Failed to wait on stack dump output.)\n");
        }
        if (status) {
          full_cerr_write("(GDB stack trace failed unexpectedly)\n");
        }
      }
    }

    int cmssw_stacktrace(void* /*arg*/) {
      set_default_signals();

      char const* const* argv = edm::service::InitRootHandlers::getPstackArgv();
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

    static constexpr char pstackName[] = "(CMSSW stack trace helper)";
    static constexpr char dashC[] = "-c";
    char InitRootHandlers::pidString_[InitRootHandlers::pidStringLength_] = {};
    char const* const InitRootHandlers::pstackArgv_[] = {pstackName, dashC, InitRootHandlers::pidString_, nullptr};
    int InitRootHandlers::parentToChild_[2] = {-1, -1};
    int InitRootHandlers::childToParent_[2] = {-1, -1};
    std::unique_ptr<std::thread> InitRootHandlers::helperThread_;
    std::unique_ptr<InitRootHandlers::ThreadTracker> InitRootHandlers::threadTracker_;
    int InitRootHandlers::stackTracePause_ = 300;
    std::vector<std::array<char, moduleBufferSize>> InitRootHandlers::moduleListBuffers_;
    std::atomic<std::size_t> InitRootHandlers::nextModule_(0), InitRootHandlers::doneModules_(0);

    InitRootHandlers::InitRootHandlers(ParameterSet const& pset, ActivityRegistry& iReg)
        : RootHandlers(),
          unloadSigHandler_(pset.getUntrackedParameter<bool>("UnloadRootSigHandler")),
          resetErrHandler_(pset.getUntrackedParameter<bool>("ResetRootErrHandler")),
          loadAllDictionaries_(pset.getUntrackedParameter<bool>("LoadAllDictionaries")),
          autoLibraryLoader_(loadAllDictionaries_ or pset.getUntrackedParameter<bool>("AutoLibraryLoader")),
          interactiveDebug_(pset.getUntrackedParameter<bool>("InteractiveDebug")) {
      stackTracePause_ = pset.getUntrackedParameter<int>("StackTracePauseTime");

      if (not threadTracker_) {
        threadTracker_ = std::make_unique<ThreadTracker>();
        iReg.watchPostEndJob([]() {
          if (threadTracker_) {
            threadTracker_->observe(false);
          }
        });
      }

      if (unloadSigHandler_) {
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
      } else if (pset.getUntrackedParameter<bool>("AbortOnSignal")) {
        cachePidInfo();

        //NOTE: ROOT can also be told to abort on these kinds of problems BUT
        // it requires an TApplication to be instantiated which causes problems
        gSystem->ResetSignal(kSigBus);
        gSystem->ResetSignal(kSigSegmentationViolation);
        gSystem->ResetSignal(kSigIllegalInstruction);
        installCustomHandler(SIGBUS, sig_dostack_then_abort);
        sigBusHandler_ = std::shared_ptr<const void>(nullptr, [](void*) { installCustomHandler(SIGBUS, sig_abort); });
        installCustomHandler(SIGSEGV, sig_dostack_then_abort);
        sigSegvHandler_ = std::shared_ptr<const void>(nullptr, [](void*) { installCustomHandler(SIGSEGV, sig_abort); });
        installCustomHandler(SIGILL, sig_dostack_then_abort);
        sigIllHandler_ = std::shared_ptr<const void>(nullptr, [](void*) { installCustomHandler(SIGILL, sig_abort); });
        installCustomHandler(SIGTERM, sig_dostack_then_abort);
        sigTermHandler_ = std::shared_ptr<const void>(nullptr, [](void*) { installCustomHandler(SIGTERM, sig_abort); });
        installCustomHandler(SIGABRT, sig_dostack_then_abort);
        sigAbrtHandler_ = std::shared_ptr<const void>(nullptr, [](void*) {
          signal(SIGABRT, SIG_DFL);  // release SIGABRT to default
        });
      }

      iReg.watchPreallocate([](edm::service::SystemBounds const& iBounds) {
        if (iBounds.maxNumberOfThreads() > moduleListBuffers_.size()) {
          moduleListBuffers_.resize(iBounds.maxNumberOfThreads());
        }
      });

      if (resetErrHandler_) {
        // Replace the Root error handler with one that uses the MessageLogger
        SetErrorHandler(RootErrorHandler);
      }

      // Enable automatic Root library loading.
      if (autoLibraryLoader_) {
        gInterpreter->SetClassAutoloading(1);
      }

      // Set ROOT parameters.
      TTree::SetMaxTreeSize(kMaxLong64);
      TH1::AddDirectory(kFALSE);
      //G__SetCatchException(0);

      // Set custom streamers
      setRefCoreStreamerInTClass();

      // Load the library containing dictionaries for std:: classes, if not already loaded.
      if (!hasDictionary(typeid(std::vector<std::vector<unsigned int>>))) {
        TypeWithDict::byName("std::vector<std::vector<unsigned int> >");
      }

      int debugLevel = pset.getUntrackedParameter<int>("DebugLevel");
      if (debugLevel > 0) {
        gDebug = debugLevel;
      }

      // Enable Root implicit multi-threading
      bool imt = pset.getUntrackedParameter<bool>("EnableIMT");
      if (imt && not ROOT::IsImplicitMTEnabled()) {
        //cmsRun uses global_control to set the number of allowed threads to use
        // we need to tell ROOT the same value in order to avoid unnecessary warnings
        ROOT::EnableImplicitMT(
            oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism));
      }
    }

    InitRootHandlers::~InitRootHandlers() {
      // close all open ROOT files
      TIter iter(gROOT->GetListOfFiles());
      TObject* obj = nullptr;
      while (nullptr != (obj = iter.Next())) {
        TFile* f = dynamic_cast<TFile*>(obj);
        if (f) {
          // We get a new iterator each time,
          // because closing a file can invalidate the iterator
          f->Close();
          iter = TIter(gROOT->GetListOfFiles());
        }
      }
      //disengage from TBB to avoid possible at exit problems
      threadTracker_.reset();
    }

    void InitRootHandlers::willBeUsingThreads() {
      //Tell Root we want to be multi-threaded
      ROOT::EnableThreadSafety();

      //When threading, also have to keep ROOT from logging all TObjects into a list
      TObject::SetObjectStat(false);

      //Have to avoid having Streamers modify themselves after they have been used
      TVirtualStreamerInfo::Optimize(false);
    }

    void InitRootHandlers::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Centralized interface to ROOT.");
      desc.addUntracked<bool>("UnloadRootSigHandler", false)
          ->setComment("If True, signals are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("ResetRootErrHandler", true)
          ->setComment(
              "If True, ROOT messages (e.g. errors, warnings) are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("AutoLibraryLoader", true)
          ->setComment("If True, enables automatic loading of data dictionaries.");
      desc.addUntracked<bool>("LoadAllDictionaries", false)->setComment("If True, loads all ROOT dictionaries.");
      desc.addUntracked<bool>("EnableIMT", true)->setComment("If True, calls ROOT::EnableImplicitMT().");
      desc.addUntracked<bool>("AbortOnSignal", true)
          ->setComment(
              "If True, do an abort when a signal occurs that causes a crash. If False, ROOT will do an exit which "
              "attempts to do a clean shutdown.");
      desc.addUntracked<bool>("InteractiveDebug", false)
          ->setComment(
              "If True, leave gdb attached to cmsRun after a crash; "
              "if False, attach gdb, print a stack trace, and quit gdb");
      desc.addUntracked<int>("DebugLevel", 0)->setComment("Sets ROOT's gDebug value.");
      desc.addUntracked<int>("StackTracePauseTime", 300)
          ->setComment("Seconds to pause other threads during stack trace.");
      descriptions.add("InitRootHandlers", desc);
    }

    char const* const* InitRootHandlers::getPstackArgv() { return pstackArgv_; }

    void InitRootHandlers::enableWarnings_() { s_ignoreWarnings = edm::RootHandlers::SeverityLevel::kInfo; }

    void InitRootHandlers::ignoreWarnings_(edm::RootHandlers::SeverityLevel level) { s_ignoreWarnings = level; }

    void InitRootHandlers::cachePidInfo() {
      if (helperThread_) {
        //Another InitRootHandlers was initialized in this job, possibly
        // because multiple EventProcessors are being used.
        //In that case, we are already all setup
        return;
      }
      std::string gdbcmd{"date; gdb -quiet -p %d"};
      if (!interactiveDebug_) {
        gdbcmd +=
            " 2>&1 <<EOF |\n"
            "set width 0\n"
            "set height 0\n"
            "set pagination no\n"
            "thread apply all bt\n"
            "EOF\n"
            "/bin/sed -n -e 's/^\\((gdb) \\)*//' -e '/^#/p' -e '/^Thread/p'";
      }
      if (snprintf(pidString_, pidStringLength_ - 1, gdbcmd.c_str(), getpid()) >= pidStringLength_) {
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
      childToParent_[0] = -1;
      childToParent_[1] = -1;
      close(parentToChild_[0]);
      close(parentToChild_[1]);
      parentToChild_[0] = -1;
      parentToChild_[1] = -1;

      if (-1 == pipe2(childToParent_, O_CLOEXEC)) {
        std::ostringstream sstr;
        sstr << "Failed to create child-to-parent pipes (errno=" << errno << "): " << strerror(errno);
        edm::Exception except(edm::errors::OtherCMS, sstr.str());
        throw except;
      }

      if (-1 == pipe2(parentToChild_, O_CLOEXEC)) {
        close(childToParent_[0]);
        close(childToParent_[1]);
        childToParent_[0] = -1;
        childToParent_[1] = -1;
        std::ostringstream sstr;
        sstr << "Failed to create child-to-parent pipes (errno=" << errno << "): " << strerror(errno);
        edm::Exception except(edm::errors::OtherCMS, sstr.str());
        throw except;
      }

      helperThread_ = std::make_unique<std::thread>(stacktraceHelperThread);
      helperThread_->detach();
    }

  }  // end of namespace service
}  // end of namespace edm

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::InitRootHandlers;
typedef edm::serviceregistry::AllArgsMaker<edm::RootHandlers, InitRootHandlers> RootHandlersMaker;
DEFINE_FWK_SERVICE_MAKER(InitRootHandlers, RootHandlersMaker);
