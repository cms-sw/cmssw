#ifndef FWCore_MessageLogger_ConfigurationHandshake_h
#define FWCore_MessageLogger_ConfigurationHandshake_h

#include "FWCore/Utilities/interface/EDMException.h"

#include <condition_variable>
#include <mutex>

namespace edm {
  class ParameterSet;

  typedef std::shared_ptr<edm::Exception> Pointer_to_new_exception_on_heap;
  typedef std::shared_ptr<Pointer_to_new_exception_on_heap> Place_for_passing_exception_ptr;

  struct ConfigurationHandshake {
    void* p;
    std::mutex m;
    std::condition_variable c;
    edm::Place_for_passing_exception_ptr epp;
    explicit ConfigurationHandshake(void* p_in, Place_for_passing_exception_ptr epp_in)
        : p(p_in), m(), c(), epp(epp_in) {}
  };
}  // namespace edm
#endif  // FWCore_MessageLogger_ConfigurationHandshake_h
