#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "ThreadAllocInfo.h"
#include "ThreadTracker.h"

namespace {
  using namespace edm::service::moduleAlloc;
  class MonitorPauseGuard {
  public:
    MonitorPauseGuard(ThreadAllocInfo& info) : info_(info) {}
    ~MonitorPauseGuard() { info_.activate(); }

  private:
    ThreadAllocInfo& info_;
  };

  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static std::any startOnThread() {
      auto& t = threadAllocInfo();
      // deactivate before creating the std::any
      // keep deactivated until the guard activates it again in ~MonitorPauseGuard
      t.deactivate();
      std::any previous = t;
      threadAllocInfo().reset();
      return previous;
    }
    static std::pair<ThreadAllocInfo, MonitorPauseGuard> stopOnThread(std::any previous) {
      auto& t = threadAllocInfo();
      t.deactivate();
      // restore state before the current measurement to allow nested measurements
      auto measured = t;
      t = std::any_cast<ThreadAllocInfo>(previous);
      assert(not t.active_);
      return {measured, t};
    }

    static ThreadAllocInfo& threadAllocInfo() {
      using namespace cms::perftools::allocMon;
      static ThreadAllocInfo s_info[ThreadTracker::kTotalEntries];
      return s_info[ThreadTracker::instance().thread_index()];
    }

  private:
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

class IntrusiveAllocMonitor : public edm::IntrusiveMonitorBase {
public:
  IntrusiveAllocMonitor() {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
  };
  ~IntrusiveAllocMonitor() override = default;

  std::any start() final { return MonitorAdaptor::startOnThread(); }
  void stop(edm::IntrusiveMonitorStackNode const* callStack, std::any previousAllocInfo) final {
    auto [info, pauseGuard] = MonitorAdaptor::stopOnThread(std::move(previousAllocInfo));
    // The pauseGuard keeps the monitoring paused during the all string operations below
    std::string name;
    {
      // concatenate the names of the call stack backwards
      edm::IntrusiveMonitorStackNode const* node = callStack;
      while (node != nullptr) {
        name = std::format("{};{}", node->name(), name);
        node = node->previousNode();
      }
      // remove the trailing semicolon
      if (not name.empty()) {
        name.pop_back();
      }
    }
    edm::LogSystem("IntrusiveAllocMonitor")
        .format("{}: requested {} added {} max alloc {} peak {} nAlloc {} nDealloc {}",
                name,
                info.requested_,
                info.presentActual_,
                info.maxSingleAlloc_,
                info.maxActual_,
                info.nAllocations_,
                info.nDeallocations_);
  }
};

typedef edm::serviceregistry::NoArgsMaker<edm::IntrusiveMonitorBase, IntrusiveAllocMonitor> IntrusiveAllocMonitorMaker;
DEFINE_FWK_SERVICE_MAKER(IntrusiveAllocMonitor, IntrusiveAllocMonitorMaker);
