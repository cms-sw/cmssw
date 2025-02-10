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
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#if defined(ALLOC_USE_PTHREADS)
#include <pthread.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#endif

#include "moduleAlloc_setupFile.h"
#include "ThreadAllocInfo.h"
#include "ThreadTracker.h"

namespace {
  using namespace cms::perftools::allocMon;
  using namespace edm::service::moduleAlloc;
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread() { threadAllocInfo().reset(); }
    static ThreadAllocInfo const& stopOnThread() {
      auto& t = threadAllocInfo();
      if (not t.active_) {
        t.reset();
      } else {
        t.deactivate();
      }
      return t;
    }

  private:
    static ThreadAllocInfo& threadAllocInfo() {
      static ThreadAllocInfo s_info[ThreadTracker::kTotalEntries];
      return s_info[ThreadTracker::instance().thread_index()];
    }
    void allocCalled(size_t iRequested, size_t iActual, void const*) final {
      auto& allocInfo = threadAllocInfo();
      if (not allocInfo.active_) {
        return;
      }
      allocInfo.nAllocations_ += 1;
      allocInfo.requested_ += iRequested;

      if (allocInfo.maxSingleAlloc_ < iRequested) {
        allocInfo.maxSingleAlloc_ = iRequested;
      }

      allocInfo.presentActual_ += iActual;
      if (allocInfo.presentActual_ > static_cast<long long>(allocInfo.maxActual_)) {
        allocInfo.maxActual_ = allocInfo.presentActual_;
      }
    }
    void deallocCalled(size_t iActual, void const*) final {
      auto& allocInfo = threadAllocInfo();
      if (not allocInfo.active_) {
        return;
      }

      allocInfo.nDeallocations_ += 1;
      allocInfo.presentActual_ -= iActual;
      if (allocInfo.presentActual_ < 0) {
        if (allocInfo.minActual_ == 0 or allocInfo.minActual_ > allocInfo.presentActual_) {
          allocInfo.minActual_ = allocInfo.presentActual_;
        }
      }
    }
  };

}  // namespace

namespace edm::service::moduleAlloc {
  Filter::Filter(std::vector<int> const* moduleIDs) : moduleIDs_{moduleIDs} {}

  bool Filter::startOnThread(int moduleID) const {
    if (not globalKeep_.load()) {
      return false;
    }
    if (keepModuleInfo(moduleID)) {
      MonitorAdaptor::startOnThread();
      return true;
    }
    return false;
  }

  const ThreadAllocInfo* Filter::stopOnThread(int moduleID) const {
    if (not globalKeep_.load()) {
      return nullptr;
    }

    if (keepModuleInfo(moduleID)) {
      return &MonitorAdaptor::stopOnThread();
    }
    return nullptr;
  }

  bool Filter::startOnThread() const {
    if (not globalKeep_.load()) {
      return false;
    }
    MonitorAdaptor::startOnThread();
    return true;
  }

  const ThreadAllocInfo* Filter::stopOnThread() const {
    if (not globalKeep_.load()) {
      return nullptr;
    }
    return &MonitorAdaptor::stopOnThread();
  }

  void Filter::setGlobalKeep(bool iShouldKeep) { globalKeep_.store(iShouldKeep); }

  bool Filter::keepModuleInfo(int moduleID) const {
    if ((nullptr == moduleIDs_) or (moduleIDs_->empty()) or
        (std::binary_search(moduleIDs_->begin(), moduleIDs_->end(), moduleID))) {
      return true;
    }
    return false;
  }
}  // namespace edm::service::moduleAlloc

class ModuleAllocMonitor {
public:
  ModuleAllocMonitor(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames")),
        nEventsToSkip_(iPS.getUntrackedParameter<unsigned int>("nEventsToSkip")),
        filter_(&moduleIDs_) {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();

    if (nEventsToSkip_ > 0) {
      filter_.setGlobalKeep(false);
    }
    if (not moduleNames_.empty()) {
      iAR.watchPreModuleConstruction([this](auto const& description) {
        auto found = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleLabel());
        if (found != moduleNames_.end()) {
          moduleIDs_.push_back(description.id());
          std::sort(moduleIDs_.begin(), moduleIDs_.end());
        }
      });

      iAR.watchPostESModuleRegistration([this](auto const& iDescription) {
        auto label = iDescription.label_;
        if (label.empty()) {
          label = iDescription.type_;
        }
        auto found = std::find(moduleNames_.begin(), moduleNames_.end(), label);
        if (found != moduleNames_.end()) {
          //NOTE: we want the id to start at 1 not 0
          moduleIDs_.push_back(-1 * (iDescription.id_ + 1));
          std::sort(moduleIDs_.begin(), moduleIDs_.end());
        }
      });
    }
    if (nEventsToSkip_ > 0) {
      iAR.watchPreSourceEvent([this](auto) {
        ++nEventsStarted_;
        if (nEventsStarted_ > nEventsToSkip_) {
          filter_.setGlobalKeep(true);
        }
      });
    }
    edm::service::moduleAlloc::setupFile(iPS.getUntrackedParameter<std::string>("fileName"), iAR, &filter_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::string>("fileName");
    ps.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>());
    ps.addUntracked<unsigned int>("nEventsToSkip", 0);
    iDesc.addDefault(ps);
  }

private:
  bool forThisModule(unsigned int iID) {
    return (moduleNames_.empty() or std::binary_search(moduleIDs_.begin(), moduleIDs_.end(), iID));
  }
  std::vector<std::string> moduleNames_;
  std::vector<int> moduleIDs_;
  unsigned int nEventsToSkip_ = 0;
  std::atomic<unsigned int> nEventsStarted_{0};
  edm::service::moduleAlloc::Filter filter_;
};

DEFINE_FWK_SERVICE(ModuleAllocMonitor);
