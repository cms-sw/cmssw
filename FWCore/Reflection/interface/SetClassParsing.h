#ifndef FWCore_Reflection_interface_SetClassParsing_h
#define FWCore_Reflection_interface_SetClassParsing_h

#include "TInterpreter.h"

#include <atomic>
#include <cassert>

namespace edm {
  /**
   * An instance of this class can be used to temporarily enable or
   * disable ROOT class parsing. The destructor restores the state.
   *
   * Because this class modifies the global state of ROOT, it must
   * not be used in concurrent context.
   */
  class SetClassParsing {
  public:
    SetClassParsing(bool enable) {
      bool expected = false;
      const bool used_concurrently_but_shouldnt = active_.compare_exchange_strong(expected, true);
      assert(used_concurrently_but_shouldnt);
      previous_ = gInterpreter->SetClassAutoparsing(enable);
    }
    ~SetClassParsing() {
      gInterpreter->SetClassAutoparsing(previous_);
      active_ = false;
    }

  private:
    int previous_;
    static std::atomic<bool> active_;  // to detect if the class is used in concurrent context
  };
}  // namespace edm

#endif
