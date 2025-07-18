#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "ThreadAllocInfo.h"
#include "ThreadTracker.h"

namespace {
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

  void start() final { MonitorAdaptor::startOnThread(); }
  void stop(std::string_view name) final {
    MonitorAdaptor::stopOnThread();
    auto& info = MonitorAdaptor::threadAllocInfo();
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
