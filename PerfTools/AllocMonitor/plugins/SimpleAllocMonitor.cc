// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     SimpleAllocMonitor
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
    void allocCalled(size_t iRequested, size_t iActual) final {
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
      nDeallocations_.fetch_add(1, std::memory_order_acq_rel);
      auto present = presentActual_.load(std::memory_order_acquire);
      if (present >= iActual) {
        presentActual_.fetch_sub(iActual, std::memory_order_acq_rel);
      }
    }

    void performanceReport() const {
      auto finalRequested = requested_.load(std::memory_order_acquire);
      auto maxActual = maxActual_.load(std::memory_order_acquire);
      auto allocs = nAllocations_.load(std::memory_order_acquire);
      auto deallocs = nDeallocations_.load(std::memory_order_acquire);

      edm::LogSystem("SimpleAllocMonitor")
          << "Memory Report"
          << "\n  [monitoring spans the lifetime of Services (first plugins made and last to be deleted)]"
          << "\n  total additional memory requested: " << finalRequested
          << "\n  max additional memory used: " << maxActual << "\n  # allocations calls:   " << allocs
          << "\n  # deallocations calls: " << deallocs;
    }

  private:
    std::atomic<size_t> requested_ = 0;
    std::atomic<size_t> presentActual_ = 0;
    std::atomic<size_t> maxActual_ = 0;
    std::atomic<size_t> nAllocations_ = 0;
    std::atomic<size_t> nDeallocations_ = 0;
  };

}  // namespace

class SimpleAllocMonitor {
public:
  SimpleAllocMonitor()
      : adaptor_(cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>()) {}

  ~SimpleAllocMonitor() {
    adaptor_->performanceReport();
    cms::perftools::AllocMonitorRegistry::instance().deregisterMonitor(adaptor_);
  }

  MonitorAdaptor* adaptor_;
};

DEFINE_FWK_SERVICE_MAKER(SimpleAllocMonitor, edm::serviceregistry::NoArgsMaker<SimpleAllocMonitor>);
