#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "AllocProfilerData.h"
#include "ThreadTracker.h"
#include "monitor_file_utilities.h"

#include <algorithm>
#include <atomic>
#include <stacktrace>
#include <string>
#include <vector>

namespace {
  using namespace cms::perftools::allocMon::profiler;

  // ---------------------------------------------------------------------------
  // MonitorAdaptor
  // ---------------------------------------------------------------------------
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread(std::string filePattern, ReportConfiguration config) {
      threadActiveMonitoring() = false;
      auto const fileCount = globalFileCounter().fetch_add(1);
      auto node = std::make_unique<MonitorStackNode>(
          "", std::move(currentMonitorStackNode()), StackNodeData(fileCount, std::move(filePattern), config));
      currentMonitorStackNode() = std::move(node);
      threadActiveMonitoring() = true;
    }

    static void stopOnThread() {
      threadActiveMonitoring() = false;
      auto node = std::move(currentMonitorStackNode());
      edm::LogSystem log("ModuleAllocProfiler");
      log.format("Ending tracing.");
      node->get().print(log, "");
      // in module-based monitoring there can't be nested measurement regions
      auto prev = node->popPreviousNode();
      assert(not prev);
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

  // ---------------------------------------------------------------------------
  // ProfilerFilter
  // ---------------------------------------------------------------------------
  class ProfilerFilter {
  public:
    ProfilerFilter(std::vector<int> const* moduleIDs, std::string filePattern, ReportConfiguration config)
        : moduleIDs_(moduleIDs), filePattern_(std::move(filePattern)), config_(config) {}

    bool startOnThread(int moduleID) const {
      if (not globalKeep_.load()) {
        return false;
      }
      if (keepModuleInfo(moduleID)) {
        MonitorAdaptor::startOnThread(filePattern_, config_);
        return true;
      }
      return false;
    }

    bool startOnThread() const {
      if (not globalKeep_.load()) {
        return false;
      }
      MonitorAdaptor::startOnThread(filePattern_, config_);
      return true;
    }

    void stopOnThread(int moduleID) const {
      if (not globalKeep_.load()) {
        return;
      }
      if (keepModuleInfo(moduleID)) {
        MonitorAdaptor::stopOnThread();
      }
    }

    void stopOnThread() const {
      if (not globalKeep_.load()) {
        return;
      }
      MonitorAdaptor::stopOnThread();
    }

    void setGlobalKeep(bool iShouldKeep) { globalKeep_.store(iShouldKeep); }

    bool keepModuleInfo(int moduleID) const {
      if ((nullptr == moduleIDs_) or (moduleIDs_->empty()) or
          (std::binary_search(moduleIDs_->begin(), moduleIDs_->end(), moduleID))) {
        return true;
      }
      return false;
    }

  private:
    mutable std::atomic<bool> globalKeep_ = true;
    std::vector<int> const* moduleIDs_;
    std::string filePattern_;
    ReportConfiguration config_;
  };

  using edm::moduleAlloc::monitor_file_utilities::module_id;

  // ---------------------------------------------------------------------------
  // setupProfilerFile
  // ---------------------------------------------------------------------------
  void setupProfilerFile(edm::ActivityRegistry& iAR, ProfilerFilter& filter) {
    // Lambdas for the three callback shapes, capturing filter by reference.
    // Each pair is used by multiple watch* calls below.
    auto const sourceStart = [&filter](auto const&) { filter.startOnThread(); };
    auto const sourceStop = [&filter](auto const&) { filter.stopOnThread(); };
    auto const edStart = [&filter](auto const&, edm::ModuleCallingContext const& mcc) {
      filter.startOnThread(static_cast<int>(module_id(mcc)));
    };
    auto const edStop = [&filter](auto const&, edm::ModuleCallingContext const& mcc) {
      filter.stopOnThread(static_cast<int>(module_id(mcc)));
    };
    auto const esStart = [&filter](auto const&, edm::ESModuleCallingContext const& mcc) {
      filter.startOnThread(-1 * static_cast<int>(module_id(mcc) + 1));
    };
    auto const esStop = [&filter](auto const&, edm::ESModuleCallingContext const& mcc) {
      filter.stopOnThread(-1 * static_cast<int>(module_id(mcc) + 1));
    };

    //NOTE: we want the id to start at 1 not 0
    iAR.watchPreESModuleConstruction(
        [&filter](auto const& iDescription) { filter.startOnThread(-1 * static_cast<int>(iDescription.id_ + 1)); });
    iAR.watchPostESModuleConstruction(
        [&filter](auto const& iDescription) { filter.stopOnThread(-1 * static_cast<int>(iDescription.id_ + 1)); });

    iAR.watchPreModuleConstruction(
        [&filter](edm::ModuleDescription const& md) { filter.startOnThread(static_cast<int>(md.id())); });
    iAR.watchPostModuleConstruction(
        [&filter](edm::ModuleDescription const& md) { filter.stopOnThread(static_cast<int>(md.id())); });

    iAR.watchPreModuleDestruction(
        [&filter](edm::ModuleDescription const& md) { filter.startOnThread(static_cast<int>(md.id())); });
    iAR.watchPostModuleDestruction(
        [&filter](edm::ModuleDescription const& md) { filter.stopOnThread(static_cast<int>(md.id())); });

    // --- Source transitions ---
    iAR.watchPreSourceConstruction(sourceStart);
    iAR.watchPostSourceConstruction(sourceStop);
    iAR.watchPreOpenFile(sourceStart);
    iAR.watchPostOpenFile(sourceStop);
    iAR.watchPreSourceEvent(sourceStart);
    iAR.watchPostSourceEvent(sourceStop);
    iAR.watchPreSourceRun(sourceStart);
    iAR.watchPostSourceRun(sourceStop);
    iAR.watchPreSourceLumi(sourceStart);
    iAR.watchPostSourceLumi(sourceStop);
    iAR.watchPreSourceNextTransition([&filter]() { filter.startOnThread(); });
    iAR.watchPostSourceNextTransition([&filter]() { filter.stopOnThread(); });
    iAR.watchPreClearEvent(sourceStart);
    iAR.watchPostClearEvent(sourceStop);

    // --- ED Module begin/end job ---
    iAR.watchPreModuleBeginJob([&filter](auto const& md) { filter.startOnThread(static_cast<int>(md.id())); });
    iAR.watchPostModuleBeginJob([&filter](auto const& md) { filter.stopOnThread(static_cast<int>(md.id())); });
    iAR.watchPreModuleEndJob([&filter](auto const& md) { filter.startOnThread(static_cast<int>(md.id())); });
    iAR.watchPostModuleEndJob([&filter](auto const& md) { filter.stopOnThread(static_cast<int>(md.id())); });

    // --- ED Module stream transitions ---
    iAR.watchPreModuleBeginStream(edStart);
    iAR.watchPostModuleBeginStream(edStop);
    iAR.watchPreModuleEndStream(edStart);
    iAR.watchPostModuleEndStream(edStop);
    iAR.watchPreModuleEvent(edStart);
    iAR.watchPostModuleEvent(edStop);
    iAR.watchPreModuleEventAcquire(edStart);
    iAR.watchPostModuleEventAcquire(edStop);
    iAR.watchPreModuleEventDelayedGet(edStart);
    iAR.watchPostModuleEventDelayedGet(edStop);
    iAR.watchPreEventReadFromSource(edStart);
    iAR.watchPostEventReadFromSource(edStop);
    iAR.watchPreModuleTransform(edStart);
    iAR.watchPostModuleTransform(edStop);
    iAR.watchPreModuleTransformAcquiring(edStart);
    iAR.watchPostModuleTransformAcquiring(edStop);
    iAR.watchPreModuleStreamBeginRun(edStart);
    iAR.watchPostModuleStreamBeginRun(edStop);
    iAR.watchPreModuleStreamEndRun(edStart);
    iAR.watchPostModuleStreamEndRun(edStop);
    iAR.watchPreModuleStreamBeginLumi(edStart);
    iAR.watchPostModuleStreamBeginLumi(edStop);
    iAR.watchPreModuleStreamEndLumi(edStart);
    iAR.watchPostModuleStreamEndLumi(edStop);

    // --- ED Module global transitions ---
    iAR.watchPreModuleBeginProcessBlock(edStart);
    iAR.watchPostModuleBeginProcessBlock(edStop);
    iAR.watchPreModuleAccessInputProcessBlock(edStart);
    iAR.watchPostModuleAccessInputProcessBlock(edStop);
    iAR.watchPreModuleEndProcessBlock(edStart);
    iAR.watchPostModuleEndProcessBlock(edStop);
    iAR.watchPreModuleGlobalBeginRun(edStart);
    iAR.watchPostModuleGlobalBeginRun(edStop);
    iAR.watchPreModuleGlobalEndRun(edStart);
    iAR.watchPostModuleGlobalEndRun(edStop);
    iAR.watchPreModuleGlobalBeginLumi(edStart);
    iAR.watchPostModuleGlobalBeginLumi(edStop);
    iAR.watchPreModuleGlobalEndLumi(edStart);
    iAR.watchPostModuleGlobalEndLumi(edStop);
    iAR.watchPreModuleWriteProcessBlock(edStart);
    iAR.watchPostModuleWriteProcessBlock(edStop);
    iAR.watchPreModuleWriteRun(edStart);
    iAR.watchPostModuleWriteRun(edStop);
    iAR.watchPreModuleWriteLumi(edStart);
    iAR.watchPostModuleWriteLumi(edStop);

    // --- ES Module transitions ---
    iAR.watchPreESModule(esStart);
    iAR.watchPostESModule(esStop);
    iAR.watchPreESModuleAcquire(esStart);
    iAR.watchPostESModuleAcquire(esStop);
  }

}  // namespace

// -----------------------------------------------------------------------------
// ModuleAllocProfiler service
// -----------------------------------------------------------------------------
class ModuleAllocProfiler {
public:
  ModuleAllocProfiler(edm::ParameterSet const& iPSet, edm::ActivityRegistry& iAR)
      : moduleNames_(iPSet.getUntrackedParameter<std::vector<std::string>>("moduleNames")),
        nEventsToSkip_(iPSet.getUntrackedParameter<unsigned int>("nEventsToSkip")),
        filePattern_(iPSet.getUntrackedParameter<std::string>("filePattern")),
        config_{.printStatistics_ = iPSet.getUntrackedParameter<bool>("statistics"),
                .deallocationReport_ = iPSet.getUntrackedParameter<bool>("deallocationReport"),
                .churnReport_ = iPSet.getUntrackedParameter<bool>("churnReport")},
        profilerFilter_(&moduleIDs_, filePattern_, config_) {
    if (moduleNames_.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "moduleNames must be non-empty: ModuleAllocProfiler is intended to profile individual modules, "
             "and profiling all modules at once would be too costly.";
    }

    if (not filePattern_.empty()) {
      if (not filePattern_.contains("%I")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%I'";
      }
      if (not filePattern_.contains("%T")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%T'";
      }
    }

    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();

    if (nEventsToSkip_ > 0) {
      profilerFilter_.setGlobalKeep(false);
    }

    iAR.watchPreModuleConstruction([this](edm::ModuleDescription const& description) {
      auto found = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleLabel());
      if (found != moduleNames_.end()) {
        moduleIDs_.push_back(static_cast<int>(description.id()));
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
        moduleIDs_.push_back(-1 * static_cast<int>(iDescription.id_ + 1));
        std::sort(moduleIDs_.begin(), moduleIDs_.end());
      }
    });

    if (nEventsToSkip_ > 0) {
      iAR.watchPreSourceEvent([this](auto) {
        ++nEventsStarted_;
        if (nEventsStarted_ > nEventsToSkip_) {
          profilerFilter_.setGlobalKeep(true);
        }
      });
    }

    setupProfilerFile(iAR, profilerFilter_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>())
        ->setComment(
            "List of ED/ES module labels to profile. Must be non-empty: the intent is to profile individual modules, "
            "and profiling all modules at once would be too costly.");
    ps.addUntracked<unsigned int>("nEventsToSkip", 0);
    ps.addUntracked<std::string>("filePattern", "")
        ->setComment(
            "Pattern for the file names for the measurement results. Must contain '%I' for the counter of different "
            "files, and '%T' for the measurement type (that are 'alloc', 'atMaxActual', 'added', 'dealloc', 'churn', "
            "'churnalloc'). If empty (default), results are printed with MessageLogger.");
    ps.addUntracked<bool>("statistics", false)
        ->setComment("Whether to print some timing statistics about the memory measurement itself. Default is false.");
    ps.addUntracked<bool>("deallocationReport", true)
        ->setComment(
            "Whether to produce a report on deallocations. On deep stack traces this can take time, so turning it off "
            "could speed up if the deallocation report is not needed.");
    ps.addUntracked<bool>("churnReport", true)
        ->setComment(
            "Whether to produce reports on memory churn. On deep stack traces this can take time, so turning it off "
            "could speed up if the churn report is not needed.");
    iDesc.addDefault(ps);
  }

private:
  std::vector<std::string> moduleNames_;
  std::vector<int> moduleIDs_;
  unsigned int nEventsToSkip_ = 0;
  std::atomic<unsigned int> nEventsStarted_{0};
  std::string filePattern_;
  cms::perftools::allocMon::profiler::ReportConfiguration config_;
  ProfilerFilter profilerFilter_;
};

DEFINE_FWK_SERVICE(ModuleAllocProfiler);

#endif  // C++ version checks
