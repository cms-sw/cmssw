// -*- C++ -*-
//
// Package:     PerfTools/MaxMemoryPreload
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

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

// Hooks the target application can be instrumented with to pause and
// unpause the MaxMemoryPreload. Pausing the monitoring during a
// multithreaded execution can result in unexpected results, because
// the setting is global.
namespace {
  std::atomic<bool> paused = false;
}
void pauseMaxMemoryPreload() { paused = true; }
void unpauseMaxMemoryPreload() { paused = false; }

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    MonitorAdaptor() noexcept = default;
    ~MonitorAdaptor() noexcept override { performanceReport(); }

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      if (paused)
        return;

      nAllocations_.fetch_add(1, std::memory_order_acq_rel);
      requested_.fetch_add(iRequested, std::memory_order_acq_rel);

      auto a = presentActual_.fetch_add(iActual, std::memory_order_acq_rel);
      a += iActual;
      auto max = maxActual_.load(std::memory_order_relaxed);

      while (a > max) {
        if (maxActual_.compare_exchange_strong(max, a, std::memory_order_acq_rel)) {
          break;
        }
      }
    }
    void deallocCalled(size_t iActual, void const*) final {
      if (paused)
        return;

      nDeallocations_.fetch_add(1, std::memory_order_acq_rel);
      auto present = presentActual_.load(std::memory_order_acquire);
      if (present >= iActual) {
        presentActual_.fetch_sub(iActual, std::memory_order_acq_rel);
      }
    }

    void performanceReport() const {
      auto finalRequested = requested_.load(std::memory_order_acquire);
      auto maxActual = maxActual_.load(std::memory_order_acquire);
      auto present = presentActual_.load(std::memory_order_acquire);
      auto allocs = nAllocations_.load(std::memory_order_acquire);
      auto deallocs = nDeallocations_.load(std::memory_order_acquire);

      std::cerr << "Memory Report: total memory requested: " << finalRequested
                << "\nMemory Report:  max memory used: " << maxActual << "\nMemory Report:  presently used: " << present
                << "\nMemory Report:  # allocations calls:   " << allocs
                << "\nMemory Report:  # deallocations calls: " << deallocs << "\n";
    }

  private:
    std::atomic<size_t> requested_ = 0;
    std::atomic<size_t> presentActual_ = 0;
    std::atomic<size_t> maxActual_ = 0;
    std::atomic<size_t> nAllocations_ = 0;
    std::atomic<size_t> nDeallocations_ = 0;
  };

  [[maybe_unused]] auto const* const s_instance =
      cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
}  // namespace
