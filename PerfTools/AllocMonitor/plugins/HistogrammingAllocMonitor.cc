// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     HistogrammingAllocMonitor
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
      auto& a = allocRequested_[bin(iRequested)];
      a.fetch_add(1, std::memory_order_acq_rel);

      auto& u = allocUsed_[bin(iActual)];
      u.fetch_add(1, std::memory_order_acq_rel);
    }
    void deallocCalled(size_t iActual) final {
      auto& u = deallocUsed_[bin(iActual)];
      u.fetch_add(1, std::memory_order_acq_rel);
    }

    void performanceReport() const {
      auto log = edm::LogSystem("HistogrammingAllocMonitor");
      log << "Memory Histogram"
          << "\n   size                 allocated           deallocated"
          << "\n                  requested      used          used";
      size_t size = 0;
      for (unsigned int i = 0; i < allocRequested_.size(); ++i) {
        log << "\n"
            << std::setw(12) << size << " " << std::setw(12) << allocRequested_[i] << " " << std::setw(12)
            << allocUsed_[i] << " " << std::setw(12) << deallocUsed_[i];
        if (size == 0) {
          size = 1;
        } else {
          size *= 2;
        }
      }
    }

  private:
    static size_t bin(size_t iValue) {
      size_t i = 0;

      while (iValue != 0) {
        ++i;
        iValue /= 2;
      }
      return i;
    };

    std::array<std::atomic<size_t>, 40> allocRequested_;
    std::array<std::atomic<size_t>, 40> allocUsed_;
    std::array<std::atomic<size_t>, 40> deallocUsed_;
  };

}  // namespace

class HistogrammingAllocMonitor {
public:
  HistogrammingAllocMonitor()
      : adaptor_(cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>()) {}

  ~HistogrammingAllocMonitor() {
    adaptor_->performanceReport();
    cms::perftools::AllocMonitorRegistry::instance().deregisterMonitor(adaptor_);
  }

  MonitorAdaptor* adaptor_;
};

DEFINE_FWK_SERVICE_MAKER(HistogrammingAllocMonitor, edm::serviceregistry::NoArgsMaker<HistogrammingAllocMonitor>);
