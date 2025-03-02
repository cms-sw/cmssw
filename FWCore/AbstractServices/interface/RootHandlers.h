#ifndef FWCore_AbstractServices_interface_RootHandlers_h
#define FWCore_AbstractServices_interface_RootHandlers_h

#include "FWCore/Utilities/interface/propagate_const.h"
namespace edm {
  class EventProcessor;
  class RootHandlers {
  public:
    enum class SeverityLevel { kInfo, kWarning, kError, kSysError, kFatal };

  private:
    struct WarningSentry {
      WarningSentry(RootHandlers* iHandler, SeverityLevel level) : m_handler(iHandler) {
        m_handler->ignoreWarnings_(level);
      };
      ~WarningSentry() { m_handler->enableWarnings_(); }
      edm::propagate_const<RootHandlers*> m_handler;
    };
    friend struct edm::RootHandlers::WarningSentry;
    friend class edm::EventProcessor;

  public:
    RootHandlers() = default;
    RootHandlers(RootHandlers const&) = delete;
    RootHandlers& operator=(RootHandlers const&) = delete;
    RootHandlers(RootHandlers&&) = delete;
    RootHandlers& operator=(RootHandlers&&) = delete;
    virtual ~RootHandlers() = default;

    template <typename F>
    void ignoreWarningsWhileDoing(F iFunc, SeverityLevel level = SeverityLevel::kWarning) {
      WarningSentry sentry(this, level);
      iFunc();
    }

  private:
    virtual void willBeUsingThreads() = 0;

    virtual void enableWarnings_() = 0;
    virtual void ignoreWarnings_(SeverityLevel level) = 0;
  };
}  // end of namespace edm

#endif
