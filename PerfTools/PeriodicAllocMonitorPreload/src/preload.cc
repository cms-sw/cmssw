// -*- C++ -*-
//
// Package:     PerfTools/PeriodicAllocMonitorPreload
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
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <charconv>

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
// unpause the PeriodicAllocMonitorPreload. Pausing the monitoring during a
// multithreaded execution can result in unexpected results, because
// the setting is global.

//NOTE: Any changes to this code may also be appropriate for PerfTools/AllocMonitor/plugins/PeriodicAllocMonitor.cc

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    MonitorAdaptor() {
      auto pid = getpid();
      std::string base = "periodic_alloc_";
      auto envFileName = std::getenv("PAM_FILENAME");
      if (envFileName) {
        base = envFileName;
      }
      std::string fileName = base + std::to_string(pid) + ".csv";

      unsigned long long interval = 1000;
      auto envInterval = std::getenv("PAM_INTERVAL_MS");
      if (envInterval) {
        (void)std::from_chars(envInterval, envInterval + strlen(envInterval), interval);
      }

      threadShutDown_ = false;
      thread_ = std::thread([this, fileName, interval]() {
        auto const start = std::chrono::steady_clock::now();
        std::ofstream fs(fileName);
        fs << "timestamp, total-requested, max-actual, "
              "present-actual, max-single, nAllocs, nDeallocs\n";
        while (continueRunning_.load()) {
          auto const now = std::chrono::steady_clock::now();
          auto reportNow = report();

          fs << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << ", "
             << reportNow.requested_ << ", " << reportNow.maxActual_ << ", " << reportNow.presentActual_ << ", "
             << reportNow.maxSingleRequested_ << ", " << reportNow.nAllocations_ << ", " << reportNow.nDeallocations_
             << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
      });
    }
    ~MonitorAdaptor() override {
      if (not threadShutDown_) {
        continueRunning_ = false;
        thread_.join();
      }
    }

    struct Report {
      size_t requested_;
      long long presentActual_;
      long long maxActual_;
      size_t nAllocations_;
      size_t nDeallocations_;
      size_t maxSingleRequested_;
    };
    Report report() const {
      Report report;
      report.requested_ = requested_.load(std::memory_order_acquire);
      report.maxActual_ = maxActual_.load(std::memory_order_acquire);
      report.presentActual_ = presentActual_.load(std::memory_order_acquire);
      report.nAllocations_ = nAllocations_.load(std::memory_order_acquire);
      report.nDeallocations_ = nDeallocations_.load(std::memory_order_acquire);
      report.maxSingleRequested_ = maxSingleRequested_.load(std::memory_order_acquire);

      return report;
    }

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      nAllocations_.fetch_add(1, std::memory_order_acq_rel);
      requested_.fetch_add(iRequested, std::memory_order_acq_rel);

      //returns previous value
      auto a = presentActual_.fetch_add(iActual, std::memory_order_acq_rel);
      a += iActual;

      auto max = maxActual_.load(std::memory_order_relaxed);
      while (a > max) {
        if (maxActual_.compare_exchange_strong(max, a, std::memory_order_acq_rel)) {
          break;
        }
      }

      auto single = maxSingleRequested_.load(std::memory_order_relaxed);
      while (iRequested > single) {
        if (maxSingleRequested_.compare_exchange_strong(single, iRequested, std::memory_order_acq_rel)) {
          break;
        }
      }
    }
    void deallocCalled(size_t iActual, void const*) final {
      if (0 == iActual)
        return;
      nDeallocations_.fetch_add(1, std::memory_order_acq_rel);
      presentActual_.fetch_sub(iActual, std::memory_order_acq_rel);
    }

    std::atomic<size_t> requested_ = 0;
    std::atomic<long long> presentActual_ = 0;
    std::atomic<long long> maxActual_ = 0;
    std::atomic<size_t> nAllocations_ = 0;
    std::atomic<size_t> nDeallocations_ = 0;
    std::atomic<size_t> maxSingleRequested_ = 0;
    std::thread thread_;
    std::atomic<bool> continueRunning_ = true;
    bool threadShutDown_ = true;
  };

  [[maybe_unused]] auto const* const s_instance =
      cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
}  // namespace
