// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     PeriodicAllocMonitor
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 15 Sep 2023 14:44:38 GMT
//

// system include files
#include <thread>
#include <chrono>
#include <fstream>

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace {
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    struct Report {
      size_t requested_;
      size_t presentActual_;
      size_t maxActual_;
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

      auto single = maxSingleRequested_.load(std::memory_order_relaxed);
      while (iRequested > single) {
        if (maxSingleRequested_.compare_exchange_strong(single, iRequested, std::memory_order_acq_rel)) {
          break;
        }
      }
    }
    void deallocCalled(size_t iActual) final {
      if (0 == iActual)
        return;
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
    std::atomic<size_t> maxSingleRequested_ = 0;
  };

}  // namespace

class PeriodicAllocMonitor {
public:
  PeriodicAllocMonitor(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR) {
    auto adaptor = cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
    auto fileName = iPS.getUntrackedParameter<std::string>("filename");
    auto interval = iPS.getUntrackedParameter<unsigned long long>("millisecondsPerMeasurement");

    threadShutDown_ = false;
    thread_ = std::thread([this, fileName, interval, adaptor]() {
      auto const start = std::chrono::steady_clock::now();
      std::ofstream fs(fileName);
      fs << "timestamp, runs-started, lumis-started, events-started, events-finished, total-requested, max-actual, "
            "present-actual, max-single, nAllocs, nDeallocs\n";
      while (continueRunning_.load()) {
        auto rStarted = nRunsStarted_.load(std::memory_order_acquire);
        auto lStarted = nLumisStarted_.load(std::memory_order_acquire);
        auto const now = std::chrono::steady_clock::now();
        auto eStarted = nEventsStarted_.load(std::memory_order_acquire);
        auto eFinished = nEventsFinished_.load(std::memory_order_acquire);
        auto report = adaptor->report();

        fs << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << ", " << rStarted << ", "
           << lStarted << ", " << eStarted << ", " << eFinished << ", " << report.requested_ << ", "
           << report.maxActual_ << ", " << report.presentActual_ << ", " << report.maxSingleRequested_ << ", "
           << report.nAllocations_ << ", " << report.nDeallocations_ << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      }
    });

    iAR.watchPreEvent([this](auto const&) { nEventsStarted_.fetch_add(1, std::memory_order_acq_rel); });
    iAR.watchPostEvent([this](auto const&) { nEventsFinished_.fetch_add(1, std::memory_order_acq_rel); });
    iAR.watchPreGlobalBeginRun([this](auto const&) { nRunsStarted_.fetch_add(1, std::memory_order_acq_rel); });
    iAR.watchPreGlobalBeginLumi([this](auto const&) { nLumisStarted_.fetch_add(1, std::memory_order_acq_rel); });
    iAR.watchPreEndJob([adaptor, this]() {
      continueRunning_ = false;
      thread_.join();
      threadShutDown_ = true;
      cms::perftools::AllocMonitorRegistry::instance().deregisterMonitor(adaptor);
    });
  }
  ~PeriodicAllocMonitor() {
    if (not threadShutDown_) {
      continueRunning_ = false;
      thread_.join();
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::string>("filename", "timing.log")->setComment("Name of file to write the reports");
    ps.addUntracked<unsigned long long>("millisecondsPerMeasurement", 1000)
        ->setComment("The frequency at which to write reports");
    iDesc.addDefault(ps);
  }

private:
  std::thread thread_;
  std::atomic<std::size_t> nRunsStarted_ = 0;
  std::atomic<std::size_t> nLumisStarted_ = 0;
  std::atomic<std::size_t> nEventsStarted_ = 0;
  std::atomic<std::size_t> nEventsFinished_ = 0;
  std::atomic<bool> continueRunning_ = true;
  bool threadShutDown_ = true;
};

DEFINE_FWK_SERVICE(PeriodicAllocMonitor);
