#ifndef FWCore_Utilities_RootHandlers_h
#define FWCore_Utilities_RootHandlers_h

#include "FWCore/Utilities/interface/propagate_const.h"
namespace edm {
  class EventProcessor;
  class RootHandlers {
  private:
    struct WarningSentry {
      WarningSentry(RootHandlers* iHandler): m_handler(iHandler){
        m_handler->ignoreWarnings_();
      };
      ~WarningSentry() {
        m_handler->enableWarnings_();
      }
      edm::propagate_const<RootHandlers*> m_handler;
    };
    friend struct edm::RootHandlers::WarningSentry;
    friend class edm::EventProcessor;

  public:
    RootHandlers () {}
    virtual ~RootHandlers () {}

    template<typename F>
    void ignoreWarningsWhileDoing(F iFunc) {
      WarningSentry sentry(this);
      iFunc();
    }
    
  private:
    virtual void willBeUsingThreads() = 0;
    
    virtual void enableWarnings_() = 0;
    virtual void ignoreWarnings_() = 0;
  };
}  // end of namespace edm

#endif // InitRootHandlers_H
