#ifndef gen_EvtGenThreadOwner_h
#define gen_EvtGenThreadOwner_h

// Pins one dedicated pthread per stream and marshals every EvtGen call onto it.
// Mirrors SimG4Core's OscarMTProducer + omt::ThreadHandoff pattern

#include "SimG4Core/Application/interface/ThreadHandoff.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

namespace gen {

  class EvtGenThreadOwner {
  public:
    explicit EvtGenThreadOwner(int stackSize) : m_handoff{stackSize} {}

    EvtGenThreadOwner(const EvtGenThreadOwner&) = delete;
    EvtGenThreadOwner& operator=(const EvtGenThreadOwner&) = delete;

    template <typename F>
    void run(F&& f) {
      auto token = edm::ServiceRegistry::instance().presentToken();
      m_handoff.runAndWait([token, &f]() {
        edm::ServiceRegistry::Operate guard{token};
        f();
      });
    }

  private:
    omt::ThreadHandoff m_handoff;
  };

} 

#endif
