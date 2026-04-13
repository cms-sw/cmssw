#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "AllocProfilerData.h"

#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include <stacktrace>

namespace {
  using namespace cms::perftools::allocMon::profiler;

  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread(std::string_view name, std::string filePattern, ReportConfiguration config) {
      threadActiveMonitoring() = false;
      auto trace = std::stacktrace::current();
      auto const fileCount = globalFileCounter().fetch_add(1);
      auto node = std::make_unique<MonitorStackNode>(
          name,
          std::move(currentMonitorStackNode()),
          StackNodeData(std::move(trace), fileCount, std::move(filePattern), config));
      {
        edm::LogSystem log("IntrusiveAllocProfiler");
        log.format("Starting tracing for \"{}\".", name);
      }
      currentMonitorStackNode() = std::move(node);
      threadActiveMonitoring() = true;
    }
    static void stopOnThread(std::string_view name) {
      threadActiveMonitoring() = false;
      {
        auto node = std::move(currentMonitorStackNode());
        edm::LogSystem log("IntrusiveAllocProfiler");
        log.format("Ending tracing for \"{}\".", name);
        node->get().print(log, name);
        currentMonitorStackNode() = node->popPreviousNode();
      }
      if (currentMonitorStackNode()) {
        threadActiveMonitoring() = true;
      }
    }

  private:
    void allocCalled(size_t iRequested, size_t iActual, void const* ptr) final {
      if (not threadActiveMonitoring()) {
        return;
      }
      // First two entries are in the internals of AllocMonitor
      constexpr auto skip = 2;
      currentMonitorStackNode()->get().recordAllocation(
          std::stacktrace::current(skip),
          AllocationRecord{.requested_ = iRequested, .actual_ = iActual, .count_ = 1},
          ptr);
    }
    void deallocCalled(size_t iActual, void const* ptr) final {
      if (not threadActiveMonitoring()) {
        return;
      }
      // First two entries are in the internals of AllocMonitor
      constexpr auto skip = 2;
      currentMonitorStackNode()->get().recordDeallocation(std::stacktrace::current(skip),
                                                          DeallocationRecord{.actual_ = iActual, .count_ = 1},
                                                          ptr,
                                                          currentMonitorStackNode()->previousNode());
    }
  };
}  // namespace

class IntrusiveAllocProfiler : public edm::IntrusiveMonitorBase {
public:
  IntrusiveAllocProfiler(edm::ParameterSet const& iPSet)
      : pattern_(iPSet.getUntrackedParameter<std::string>("filePattern")),
        config_{.printStatistics_ = iPSet.getUntrackedParameter<bool>("statistics"),
                .deallocationReport_ = iPSet.getUntrackedParameter<bool>("deallocationReport"),
                .churnReport_ = iPSet.getUntrackedParameter<bool>("churnReport")} {
    if (not pattern_.empty()) {
      if (not pattern_.contains("%I")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%I'";
      }
      if (not pattern_.contains("%T")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%T'";
      }
    }

    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
  };
  ~IntrusiveAllocProfiler() noexcept override = default;

  void start(std::string_view name, bool nameIsString) final { MonitorAdaptor::startOnThread(name, pattern_, config_); }
  void stop(std::string_view name) noexcept final { MonitorAdaptor::stopOnThread(name); }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::string>("filePattern", "")
        ->setComment(
            "Pattern for the file names for the measurement results. Must contain '%I' for the counter of different "
            "files, and '%T' for the measurement type (that are 'alloc', 'dealloc', 'atMaxAlloc', 'added', 'churn', "
            "'churnalloc'). If empty (default), results are printed with MessageLogger.");
    ps.addUntracked<bool>("statistics", false)
        ->setComment("Whether to print some timing statistics about the memory measurement itself. Default is false.");
    ps.addUntracked<bool>("deallocationReport", true)
        ->setComment(
            "Whether to produce a report on deallocations. On deep stack traces this can take time, so turning it off "
            "could speed up if the deallocation report is not needed");
    ps.addUntracked<bool>("churnReport", true)
        ->setComment(
            "Whether to produce reports on memory churn. On deep stack traces this can take time, so turning it off "
            "could speed up if the deallocation report is not needed");
    iDesc.addDefault(ps);
  }

private:
  std::string pattern_;
  cms::perftools::allocMon::profiler::ReportConfiguration const config_;
};

typedef edm::serviceregistry::ParameterSetMaker<edm::IntrusiveMonitorBase, IntrusiveAllocProfiler>
    IntrusiveAllocProfilerMaker;
DEFINE_FWK_SERVICE_MAKER(IntrusiveAllocProfiler, IntrusiveAllocProfilerMaker);

#endif  // C++ version checks
