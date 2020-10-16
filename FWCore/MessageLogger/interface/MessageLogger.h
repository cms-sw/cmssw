#ifndef MessageLogger_MessageLogger_h
#define MessageLogger_MessageLogger_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     <none>
// Functions:   LogSystem,   LogError,   LogWarning, LogInfo,     LogDebug
//              LogAbsolute, LogProblem, LogPrint,   LogVerbatim, LogTrace
//			     LogImportant
//

//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
//     Major Split:  Tue Feb 14 11:00:00 CST 2006
//		     See MessageService/interface/MessageLogger.h
//
// =================================================

// system include files

#include <memory>
#include <string>

// user include files

// forward declarations

#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  namespace level {
    struct System {
      static constexpr const ELseverityLevel level = ELsevere;
      constexpr static bool suppress() noexcept { return false; }
    };
    struct Error {
      static constexpr const ELseverityLevel level = ELerror;
      static bool suppress() noexcept { return !MessageDrop::instance()->errorEnabled; }
    };
    struct Warning {
      static constexpr const ELseverityLevel level = ELwarning;
      static bool suppress() noexcept {
        return (MessageDrop::warningAlwaysSuppressed || !MessageDrop::instance()->warningEnabled);
      }
    };
    struct FwkInfo {
      static constexpr const ELseverityLevel level = ELfwkInfo;
      static bool suppress() noexcept {
        return (MessageDrop::fwkInfoAlwaysSuppressed || !MessageDrop::instance()->fwkInfoEnabled);
      }
    };
    struct Info {
      static constexpr const ELseverityLevel level = ELinfo;
      static bool suppress() noexcept {
        return (MessageDrop::infoAlwaysSuppressed || !MessageDrop::instance()->infoEnabled);
      }
    };
    struct Debug {
      static constexpr const ELseverityLevel level = ELdebug;
      constexpr static bool suppress() noexcept { return false; }
    };
  }  // namespace level

  template <typename LVL, bool VERBATIM>
  class Log {
  public:
    using ThisLog = Log<LVL, VERBATIM>;
    explicit Log(std::string_view id) : ap(LVL::level, id, VERBATIM, LVL::suppress()) {}
    Log(ThisLog&&) = default;
    Log(ThisLog const&) = delete;
    Log& operator=(ThisLog const&) = delete;
    Log& operator=(ThisLog&&) = default;
    ~Log() = default;

    template <class T>
    ThisLog& operator<<(T const& t) {
      if (ap.valid())
        ap << t;
      return *this;
    }
    ThisLog& operator<<(std::ostream& (*f)(std::ostream&)) {
      if (ap.valid())
        ap << f;
      return *this;
    }
    ThisLog& operator<<(std::ios_base& (*f)(std::ios_base&)) {
      if (ap.valid())
        ap << f;
      return *this;
    }

    template <typename... Args>
    ThisLog& format(std::string_view fmt, Args const&... args) {
      if (ap.valid())
        ap.format(fmt, args...);
      return *this;
    }

    template <typename F>
    ThisLog& log(F&& iF) {
      if (ap.valid()) {
        iF(ap);
      }
      return *this;
    }

  protected:
    Log() = default;
    //Want standard copy ctr to be deleted to make compiler errors
    // clearer. This does the same thing but with different signature
    //Needed for LogDebug and LogTrace macros
    Log(std::nullptr_t, ThisLog const& iOther) : ap(iOther.ap) {}

  private:
    MessageSender ap;
  };
  using LogWarning = Log<level::Warning, false>;
  using LogError = Log<level::Error, false>;
  using LogSystem = Log<level::System, false>;
  using LogInfo = Log<level::Info, false>;
  using LogFwkInfo = Log<level::FwkInfo, false>;

  using LogVerbatim = Log<level::Info, true>;
  using LogFwkVerbatim = Log<level::FwkInfo, true>;
  using LogPrint = Log<level::Warning, true>;
  using LogProblem = Log<level::Error, true>;
  // less judgemental verbatim version of LogError
  using LogImportant = Log<level::Error, true>;
  using LogAbsolute = Log<level::System, true>;

  void LogStatistics();

  class LogDebug_ : public Log<level::Debug, false> {
  public:
    LogDebug_() = default;
    explicit LogDebug_(std::string_view id, std::string_view file, int line);
    //Needed for the LogDebug macro
    LogDebug_(Log<level::Debug, false> const& iOther) : Log<level::Debug, false>(nullptr, iOther) {}

  private:
    std::string_view stripLeadingDirectoryTree(std::string_view file) const;
  };  // LogDebug_

  class LogTrace_ : public Log<level::Debug, true> {
  public:
    LogTrace_() = default;
    explicit LogTrace_(std::string_view id) : Log<level::Debug, true>(id) {}
    //Needed for the LogTrace macro
    LogTrace_(Log<level::Debug, true> const& iOther) : Log<level::Debug, true>(nullptr, iOther) {}
  };

  namespace impl {
    //Needed for LogDebug and LogTrace macros in order to get the
    // type on both sides of the ?: to be the same
    struct LogDebugAdapter {
      //Need an operator with lower precendence than operator<<
      LogDebug_ operator|(Log<level::Debug, false>& iOther) { return LogDebug_(iOther); }
      LogTrace_ operator|(Log<level::Debug, true>& iOther) { return LogTrace_(iOther); }
    };
  }  // namespace impl

  namespace edmmltest {
    struct WarningThatSuppressesLikeLogInfo {
      static constexpr const ELseverityLevel level = ELwarning;
      static bool suppress() noexcept {
        return (MessageDrop::infoAlwaysSuppressed || !MessageDrop::instance()->warningEnabled);
      }
    };

    using LogWarningThatSuppressesLikeLogInfo = Log<WarningThatSuppressesLikeLogInfo, false>;
  }  // end namespace edmmltest

  class Suppress_LogDebug_ {
    // With any decent optimization, use of Suppress_LogDebug_ (...)
    // including streaming of items to it via operator<<
    // will produce absolutely no executable code.
  public:
    template <class T>
    Suppress_LogDebug_& operator<<(T const&) {
      return *this;
    }
    Suppress_LogDebug_& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
    Suppress_LogDebug_& operator<<(std::ios_base& (*)(std::ios_base&)) { return *this; }

    template <typename... Args>
    Suppress_LogDebug_& format(std::string_view fmt, Args const&... args) {
      return *this;
    }

    template <typename F>
    Suppress_LogDebug_& log(F&& iF) {
      return *this;
    }
  };  // Suppress_LogDebug_

  bool isDebugEnabled();
  bool isInfoEnabled();
  bool isFwkInfoEnabled();
  bool isWarningEnabled();
  void HaltMessageLogging();
  void FlushMessageLog();
  void clearMessageLog();
  void GroupLogStatistics(std::string_view category);
  bool isMessageProcessingSetUp();

  // The following two methods have no effect except in stand-alone apps
  // that do not create a MessageServicePresence:
  void setStandAloneMessageThreshold(edm::ELseverityLevel const& severity);
  void squelchStandAloneMessageCategory(std::string const& category);

}  // namespace edm

// The preprocessor symbol controlling suppression of LogDebug is EDM_ML_DEBUG.  Thus by default LogDebug is
// If LogDebug is suppressed, all code past the LogDebug(...) is squelched.
// See doc/suppression.txt.

#ifndef EDM_ML_DEBUG
#define LogDebug(id) true ? edm::Suppress_LogDebug_() : edm::Suppress_LogDebug_()
#define LogTrace(id) true ? edm::Suppress_LogDebug_() : edm::Suppress_LogDebug_()
#else
#define LogDebug(id)                                                                       \
  (edm::MessageDrop::debugAlwaysSuppressed || !edm::MessageDrop::instance()->debugEnabled) \
      ? edm::LogDebug_()                                                                   \
      : edm::impl::LogDebugAdapter() | edm::LogDebug_(id, __FILE__, __LINE__)
#define LogTrace(id)                                                                       \
  (edm::MessageDrop::debugAlwaysSuppressed || !edm::MessageDrop::instance()->debugEnabled) \
      ? edm::LogTrace_()                                                                   \
      : edm::impl::LogDebugAdapter() | edm::LogTrace_(id)
#endif

// These macros reduce the need to pollute the code with #ifdefs. The
// idea is that the condition is checked only if debugging is enabled.
// That way the condition expression may use variables that are
// declared only if EDM_ML_DEBUG is enabled. If it is disabled, rely
// on the fact that LogDebug/LogTrace should compile to no-op.
#ifdef EDM_ML_DEBUG
#define IfLogDebug(cond, cat) \
  if (cond)                   \
  LogDebug(cat)
#define IfLogTrace(cond, cat) \
  if (cond)                   \
  LogTrace(cat)
#else
#define IfLogDebug(cond, cat) LogDebug(cat)
#define IfLogTrace(cond, cat) LogTrace(cat)
#endif

#endif  // MessageLogger_MessageLogger_h
