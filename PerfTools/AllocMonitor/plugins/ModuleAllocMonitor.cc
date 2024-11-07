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

namespace {
  inline auto thread_id() {
#if defined(ALLOC_USE_PTHREADS)
    /*NOTE: if use pthread_self, the values returned by linux had
     lots of hash collisions when using a simple % hash. Worked
     better if first divided value by 0x700 and then did %. 
     [test done on el8]
 */
    return pthread_self();
#else
    return syscall(SYS_gettid);
#endif
  }

  struct ThreadTracker {
    static constexpr unsigned int kHashedEntries = 128;
    static constexpr unsigned int kExtraEntries = 128;
    static constexpr unsigned int kTotalEntries = kHashedEntries + kExtraEntries;
    using entry_type = decltype(thread_id());
    static constexpr entry_type kUnusedEntry = ~entry_type(0);
    std::array<std::atomic<entry_type>, kHashedEntries> hashed_threads_;
    std::array<std::atomic<entry_type>, kExtraEntries> extra_threads_;

    ThreadTracker() {
      //put a value which will not match the % used when looking up the entry
      entry_type entry = 0;
      for (auto& v : hashed_threads_) {
        v = ++entry;
      }
      //assume kUsedEntry is not a valid thread-id
      for (auto& v : extra_threads_) {
        v = kUnusedEntry;
      }
    }

    std::size_t thread_index() {
      auto id = thread_id();
      auto index = thread_index_guess(id);
      auto used_id = hashed_threads_[index].load();

      if (id == used_id) {
        return index;
      }
      //try to be first thread to grab the index
      auto expected = entry_type(index + 1);
      if (used_id == expected) {
        if (hashed_threads_[index].compare_exchange_strong(expected, id)) {
          return index;
        } else {
          //another thread just beat us so have to go to non-hash storage
          return find_new_index(id);
        }
      }
      //search in non-hash storage
      return find_index(id);
    }

  private:
    std::size_t thread_index_guess(entry_type id) const {
#if defined(ALLOC_USE_PTHREADS)
      return (id / 0x700) % kHashedEntries;
#else
      return id % kHashedEntries;
#endif
    }

    std::size_t find_new_index(entry_type id) {
      std::size_t index = 0;
      for (auto& v : extra_threads_) {
        entry_type expected = kUnusedEntry;
        if (v == expected) {
          if (v.compare_exchange_strong(expected, id)) {
            return index + kHashedEntries;
          }
        }
        ++index;
      }
      //failed to find an open entry
      abort();
      return 0;
    }

    std::size_t find_index(entry_type id) {
      std::size_t index = 0;
      for (auto const& v : extra_threads_) {
        if (v == id) {
          return index + kHashedEntries;
        }
        ++index;
      }
      return find_new_index(id);
    }
  };

  static ThreadTracker& getTracker() {
    static ThreadTracker s_tracker;
    return s_tracker;
  }

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
      return s_info[getTracker().thread_index()];
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
