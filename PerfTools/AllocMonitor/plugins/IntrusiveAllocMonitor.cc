#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "ThreadAllocInfo.h"
#include "ThreadTracker.h"

namespace {
  using namespace edm::service::moduleAlloc;

  /**
   * These objects form a linked list of nested uses
   * IntrusiveAllocMonitor measurements.
   */
  class MonitorStackNode {
  public:
    MonitorStackNode(std::string_view name,
                     bool nameIsString,
                     ThreadAllocInfo const& previousInfo,
                     std::unique_ptr<MonitorStackNode> previousNode)
        : name_(name), previousInfo_(previousInfo), previousNode_(std::move(previousNode)) {
      if (nameIsString and previousNode_) {
        auto& n = previousNode_->nestedNameSizes_;
        n.sum_ += name.size();
        n.count_ += 1;
      }
    }
    MonitorStackNode(MonitorStackNode const&) = delete;
    MonitorStackNode& operator=(MonitorStackNode const&) = delete;
    MonitorStackNode(MonitorStackNode&&) = delete;
    MonitorStackNode& operator=(MonitorStackNode&&) = delete;

    ~MonitorStackNode() noexcept = default;

    std::string_view name() const { return name_; }
    ThreadAllocInfo const& previousAllocInfo() const { return previousInfo_; }
    MonitorStackNode const* previousNode() const { return previousNode_.get(); }

    struct NestedNameSizes {
      size_t sum_ = 0;
      size_t count_ = 0;
    };
    NestedNameSizes const& nestedNameSizes() const { return nestedNameSizes_; }

    std::unique_ptr<MonitorStackNode> popPreviousNode() { return std::move(previousNode_); }

  private:
    std::string_view name_;
    NestedNameSizes nestedNameSizes_;
    ThreadAllocInfo previousInfo_;
    std::unique_ptr<MonitorStackNode> previousNode_;
  };
  std::unique_ptr<MonitorStackNode>& currentMonitorStackNode() {
    static thread_local std::unique_ptr<MonitorStackNode> ptr;
    return ptr;
  }

  class PreviousStateRestoreGuard {
  public:
    PreviousStateRestoreGuard(std::unique_ptr<MonitorStackNode> node, ThreadAllocInfo& info)
        : currentNode_(std::move(node)), info_(info) {}
    ~PreviousStateRestoreGuard() noexcept {
      currentMonitorStackNode() = currentNode_->popPreviousNode();
      info_ = currentNode_->previousAllocInfo();
      assert(not info_.active_);

      // deallocate outside of measurement
      currentNode_.reset();

      info_.activate();
    }

    ThreadAllocInfo const& currentAllocInfo() const { return info_; }
    MonitorStackNode const* currentNode() const { return currentNode_.get(); }

  private:
    std::unique_ptr<MonitorStackNode> currentNode_;
    ThreadAllocInfo& info_;
  };

  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread(std::string_view name, bool nameIsString) {
      auto& t = threadAllocInfo();
      // deactivate before allocating the MonitorStackNode
      // keep the previous measurement deactivated until the guard activates it again in ~PreviousStateRestoreGuard
      t.deactivate();
      // push a node to the top of the MonitorStackNode list
      auto node = std::make_unique<MonitorStackNode>(name, nameIsString, t, std::move(currentMonitorStackNode()));
      currentMonitorStackNode() = std::move(node);
      t.reset();
    }
    static PreviousStateRestoreGuard stopOnThread() {
      auto& t = threadAllocInfo();
      t.deactivate();
      // pop the top node from the MonitorStackNode list
      return {std::move(currentMonitorStackNode()), t};
    }

  private:
    static ThreadAllocInfo& threadAllocInfo() {
      using namespace cms::perftools::allocMon;
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

class IntrusiveAllocMonitor : public edm::IntrusiveMonitorBase {
public:
  IntrusiveAllocMonitor() {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
  };
  ~IntrusiveAllocMonitor() noexcept override = default;

  void start(std::string_view name, bool nameIsString) final { MonitorAdaptor::startOnThread(name, nameIsString); }
  void stop(std::string_view nameArg) noexcept final {
    // If an exception is thrown here, can't do much more than ignore it
    CMS_SA_ALLOW try {
      auto guard = MonitorAdaptor::stopOnThread();
      // The guard keeps the monitoring paused during the all string operations below
      edm::LogSystem log("IntrusiveAllocMonitor");
      auto const& info = guard.currentAllocInfo();
      log.format("measured: requested {} added {} max alloc {} peak {} nAlloc {} nDealloc {}",
                 info.requested_,
                 info.presentActual_,
                 info.maxSingleAlloc_,
                 info.maxActual_,
                 info.nAllocations_,
                 info.nDeallocations_);

      MonitorStackNode const* node = guard.currentNode();
      if (node != nullptr) {
        // sanity check
        assert(nameArg == node->name());
      }
      int depth = 0;
      while (node != nullptr) {
        log.format("\n[{}] {}", depth, node->name());
        node = node->previousNode();
        ++depth;
      }
      auto const& nestedNames = guard.currentNode()->nestedNameSizes();
      if (nestedNames.count_ > 0) {
        log.format("\nThis includes at least {} bytes in {} allocations from string names in nested measurements",
                   nestedNames.sum_,
                   nestedNames.count_);
      }
    } catch (...) {
    }
  }
};

typedef edm::serviceregistry::NoArgsMaker<edm::IntrusiveMonitorBase, IntrusiveAllocMonitor> IntrusiveAllocMonitorMaker;
DEFINE_FWK_SERVICE_MAKER(IntrusiveAllocMonitor, IntrusiveAllocMonitorMaker);
