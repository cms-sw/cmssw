// -*- C++ -*-
//
// Package:     PerfTools/ThresholdAbortPreload
// Class  :     preload
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Aug 2023 17:56:44 GMT
//

// system include files
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <charconv>
#include <format>

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static volatile std::atomic<int> dummy = 0;
extern "C" {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, optimize("O0")))
#endif
void break_threshold_abort_alloc_monitor() {
  dummy = 1;
}
}

namespace {
  template <typename T>
  void fromEnv(const char* iEnv, T& oValue) {
    auto env = std::getenv(iEnv);
    if (env) {
      std::from_chars(env, env + strlen(env), oValue);
    }
  }

  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    MonitorAdaptor() : m_skip{0}, m_min{0}, m_max{0}, m_count{0} {
      fromEnv("TAAM_SKIP", m_skip);
      fromEnv("TAAM_MIN", m_min);
      fromEnv("TAAM_MAX", m_max);
      if (std::getenv("TAAM_BREAK")) {
        m_break = true;
      }
    }

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      if (iRequested >= m_min) {
        if (m_max == 0 or iRequested <= m_max) {
          auto v = ++m_count;
          if (v > m_skip) {
            if (m_break) {
              std::cout << std::format(
                               "break_threshold_abort_alloc_monitor: abort threshold reached count: {} request: {}",
                               v,
                               iRequested)
                        << std::endl;
              break_threshold_abort_alloc_monitor();
            } else {
              std::cout << std::format("abort threshold reached count: {} request: {}", v, iRequested) << std::endl;
              abort();
            }
          }
        }
      }
    }
    void deallocCalled(size_t iActual, void const*) final {}

    unsigned int m_skip;
    size_t m_min;
    size_t m_max;
    std::atomic<unsigned int> m_count;
    bool m_break = false;
  };

  [[maybe_unused]] auto const* const s_instance =
      cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
}  // namespace
