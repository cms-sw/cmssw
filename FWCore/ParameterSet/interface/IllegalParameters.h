#ifndef FWCore_PArameterSet_IllegalParameters_h
#define FWCore_PArameterSet_IllegalParameters_h

#include <atomic>

namespace edm {

  class EventProcessor;
  class SubProcess;
  class ParameterSetDescription;

  class IllegalParameters {
  private:
    static std::atomic<bool> throwAnException_;
    static bool throwAnException() { return throwAnException_; }
    static void setThrowAnException(bool v) { throwAnException_ = v; }

    friend class EventProcessor;
    friend class SubProcess;
    friend class ParameterSetDescription;
  };
}

#endif
