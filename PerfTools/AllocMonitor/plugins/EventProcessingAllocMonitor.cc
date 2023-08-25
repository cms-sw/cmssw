// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     EventProcessingAllocMonitor
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 21 Aug 2023 20:31:57 GMT
//

// system include files
#include <atomic>

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    void performanceReport() {
      started_.store(false, std::memory_order_release);

      auto finalRequested = requested_.load(std::memory_order_acquire);
      auto maxActual = maxActual_.load(std::memory_order_acquire);
      auto present = presentActual_.load(std::memory_order_acquire);
      auto allocs = nAllocations_.load(std::memory_order_acquire);
      auto deallocs = nDeallocations_.load(std::memory_order_acquire);

      edm::LogSystem("EventProcessingAllocMonitor")
          << "Event Processing Memory Report"
          << "\n  total memory requested: " << finalRequested << "\n  max memory used: " << maxActual
          << "\n  total memory not deallocated: " << present << "\n  # allocations calls:   " << allocs
          << "\n  # deallocations calls: " << deallocs;
    }

    void start() { started_.store(true, std::memory_order_release); }

  private:
    void allocCalled(size_t iRequested, size_t iActual) final {
      if (not started_.load(std::memory_order_acquire)) {
        return;
      }
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
    }
    void deallocCalled(size_t iActual) final {
      if (not started_.load(std::memory_order_acquire)) {
        return;
      }
      nDeallocations_.fetch_add(1, std::memory_order_acq_rel);
      auto present = presentActual_.load(std::memory_order_acquire);
      if (present >= iActual) {
        presentActual_.fetch_sub(iActual, std::memory_order_acq_rel);
      }
    }

    std::atomic<size_t> requested_ = 0;
    std::atomic<size_t> presentActual_ = 0;
    std::atomic<size_t> maxActual_ = 0;
    std::atomic<size_t> nAllocations_ = 0;
    std::atomic<size_t> nDeallocations_ = 0;

    std::atomic<bool> started_ = false;
  };

}  // namespace

class EventProcessingAllocMonitor {
public:
  EventProcessingAllocMonitor(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR) {
    auto adaptor = cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
    ;
    iAR.postBeginJobSignal_.connect([adaptor]() { adaptor->start(); });
    iAR.preEndJobSignal_.connect([adaptor]() {
      adaptor->performanceReport();
      cms::perftools::AllocMonitorRegistry::instance().deregisterMonitor(adaptor);
    });
  }
};

DEFINE_FWK_SERVICE(EventProcessingAllocMonitor);
