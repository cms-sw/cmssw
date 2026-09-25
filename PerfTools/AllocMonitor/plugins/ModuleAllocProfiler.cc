#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "AllocProfilerCommon.h"
#include "ThreadTracker.h"
#include "monitor_file_utilities.h"

#include <boost/algorithm/string.hpp>

#include <oneapi/tbb/concurrent_unordered_map.h>

#include <algorithm>
#include <atomic>
#include <optional>
#include <stacktrace>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
  constexpr int kSourceModuleID = 0;

  using namespace cms::perftools::allocMon::profiler;

  // ---------------------------------------------------------------------------
  // MonitorAdaptor
  // ---------------------------------------------------------------------------
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread(std::string filePattern, std::string_view measurementName, ReportConfiguration config) {
      threadActiveMonitoring() = false;
      auto node = std::make_unique<MonitorStackNode>(
          measurementName, std::move(currentMonitorStackNode()), StackNodeData(std::move(filePattern), config));
      currentMonitorStackNode() = std::move(node);
      threadActiveMonitoring() = true;
    }

    static void stopOnThread() {
      threadActiveMonitoring() = false;
      auto node = std::move(currentMonitorStackNode());
      edm::LogSystem log("ModuleAllocProfiler");
      log.format("Ending tracing.");
      node->get().print(log, node->name());
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
    ProfilerFilter(std::vector<int> const& moduleIDs,
                   std::size_t nModules,
                   std::string filePattern,
                   ReportConfiguration config)
        : moduleIDs_(moduleIDs), filePattern_(std::move(filePattern)), config_(config) {
      // unique_ptrs of arrays instead of vectors because atomic is non-copyable
      accessInputProcessBlockCounters_ = std::make_unique<std::atomic<unsigned int>[]>(nModules);
      esModuleCallCounters_ =
          std::make_unique<oneapi::tbb::concurrent_unordered_map<std::uintptr_t, std::atomic<unsigned int>>[]>(
              nModules);
      esPendingAcquireCounts_ =
          std::make_unique<oneapi::tbb::concurrent_unordered_map<std::uintptr_t, std::atomic<int>>[]>(nModules);
    }

    // Start monitoring on this thread.
    // label is the module label (for %M substitution).
    // signal is the activity signal name (for %S substitution).
    // count is the per-(module,signal) occurrence counter (for %I substitution).
    // callID is the ESModule callID (for %C substitution); 0 for non-ES signals.
    // measurementName is forwarded to StackNodeData::print() as the file comment (e.g. record type for ES modules).
    // Callers are responsible for checking tryCompactIndex before calling this.
    void startOnThread(std::string_view label,
                       std::string_view signal,
                       unsigned int count,
                       std::string_view measurementName = "",
                       std::uintptr_t callID = 0) const {
      auto filePattern = filePattern_;  // copy the template
      boost::replace_all(filePattern, "%M", label);
      boost::replace_all(filePattern, "%S", signal);
      boost::replace_all(filePattern, "%I", std::to_string(count));
      boost::replace_all(filePattern, "%C", std::to_string(callID));

      MonitorAdaptor::startOnThread(std::move(filePattern), measurementName, config_);
    }

    void stopOnThread(int moduleID) const {
      if (not globalKeep_.load()) {
        return;
      }
      if (keepModuleInfo(moduleID)) {
        MonitorAdaptor::stopOnThread();
      }
    }

    // for ClearEvemt
    void stopOnThread() const {
      if (not globalKeep_.load()) {
        return;
      }
      MonitorAdaptor::stopOnThread();
    }

    void setGlobalKeep(bool iShouldKeep) { globalKeep_.store(iShouldKeep); }

    bool keepModuleInfo(int moduleID) const {
      return std::binary_search(moduleIDs_.begin(), moduleIDs_.end(), moduleID);
    }

    std::size_t compactIndex(int moduleID) const {
      return static_cast<std::size_t>(std::lower_bound(moduleIDs_.begin(), moduleIDs_.end(), moduleID) -
                                      moduleIDs_.begin());
    }

    // Checks globalKeep_ and module membership in one step (one binary search).
    // Returns the compact index if monitoring should proceed, nullopt otherwise.
    std::optional<std::size_t> tryCompactIndex(int moduleID) const {
      if (not globalKeep_.load()) {
        return std::nullopt;
      }
      auto it = std::lower_bound(moduleIDs_.begin(), moduleIDs_.end(), moduleID);
      if (it == moduleIDs_.end() || *it != moduleID) {
        return std::nullopt;
      }
      return static_cast<std::size_t>(it - moduleIDs_.begin());
    }

    void allocateCounters(unsigned int nStreams, unsigned int nLumis, unsigned int nRuns) {
      event_.slots.resize(nStreams, 0u);
      lumi_.slots.resize(nLumis, 0u);
      run_.slots.resize(nRuns, 0u);
    }

    // Returns the current call counter for (esModuleID, callID), then increments it.
    // Generally it is possible to have different callbacks of the same ESModule to be called concurrently, so we need
    // to keep thread safety in mind (hence concurrent_unordered_map and atomic counters). In practice the ESModule
    // callbacks do not occur that often.
    unsigned int getAndIncrementESCounter(std::size_t idx, std::uintptr_t callID) {
      auto& map = esModuleCallCounters_[idx];
      auto found = map.find(callID);
      if (found != map.end()) {
        return found->second++;
      }
      map.emplace(callID, 1u);
      return 0u;
    }

    // For ESModuleAcquire: increments the counter and stores the count as pending
    // so the paired ESModule call can reuse the same count.
    unsigned int getAndStoreESAcquireCounter(std::size_t idx, std::uintptr_t callID) {
      unsigned int count = getAndIncrementESCounter(idx, callID);
      auto& pending = esPendingAcquireCounts_[idx];
      auto result = pending.emplace(callID, static_cast<int>(count));
      if (!result.second) {
        result.first->second.store(static_cast<int>(count));
      }
      return count;
    }

    // For ESModule: reuses the count from a prior ESModuleAcquire for the same callID if available,
    // otherwise increments the counter as usual.
    unsigned int getESModuleCounter(std::size_t idx, std::uintptr_t callID) {
      auto& pending = esPendingAcquireCounts_[idx];
      auto found = pending.find(callID);
      if (found != pending.end()) {
        int val = found->second.exchange(-1);
        if (val >= 0) {
          return static_cast<unsigned int>(val);
        }
      }
      return getAndIncrementESCounter(idx, callID);
    }

    // Slots for concurrent transitions (events, lumis, runs) and a global counter for each
    struct TransitionCounters {
      std::vector<unsigned int> slots;
      std::atomic<unsigned int> global{0};
    };
    TransitionCounters event_;  // slots sized [maxNumberOfStreams]
    TransitionCounters lumi_;   // slots sized [maxNumberOfConcurrentLuminosityBlocks]
    TransitionCounters run_;    // slots sized [maxNumberOfConcurrentRuns]

    // Per-module counters for moduleAccessInputProcessBlock, indexed by compactIndex(moduleID).
    // This wastes a little bit of memory for ESModules (for which there will be an entry) and EDModules that do not use
    // this transition, but simplifies the implementation a lot
    std::unique_ptr<std::atomic<unsigned int>[]> accessInputProcessBlockCounters_;

    // Per-ES-module call counters keyed by callID, indexed by compactIndex(esModuleID), keyed by callID.
    // This wastes a little bit of memory for EDModules (for which there will be an entry) but simplifies the
    // implementation a lot.
    std::unique_ptr<oneapi::tbb::concurrent_unordered_map<std::uintptr_t, std::atomic<unsigned int>>[]>
        esModuleCallCounters_;

    // Pending acquire counts for ESModuleAcquire+ESModule pairs, indexed by compactIndex(esModuleID), keyed by callID.
    // ESModuleAcquire stores the count here (non-negative); ESModule exchanges it to -1 to reuse the same count.
    std::unique_ptr<oneapi::tbb::concurrent_unordered_map<std::uintptr_t, std::atomic<int>>[]> esPendingAcquireCounts_;

  private:
    mutable std::atomic<bool> globalKeep_ = true;
    std::vector<int> const& moduleIDs_;
    std::string filePattern_;
    ReportConfiguration config_;
  };

  using edm::moduleAlloc::monitor_file_utilities::module_id;

  // ---------------------------------------------------------------------------
  // setupProfilerFile
  // ---------------------------------------------------------------------------
  void setupProfilerFile(edm::ActivityRegistry& iAR, ProfilerFilter& filter, const bool profileClearEvent) {
    // Stop lambdas
    // Generic ED stop (used by all StreamContext/GlobalContext + ModuleCallingContext signals):
    auto edStop = [&filter](auto const&, edm::ModuleCallingContext const& mcc) {
      filter.stopOnThread(static_cast<int>(module_id(mcc)));
    };
    // Generic ES stop:
    auto esStop = [&filter](auto const&, edm::ESModuleCallingContext const& mcc) {
      filter.stopOnThread(-1 * static_cast<int>(module_id(mcc) + 1));
    };
    // MD stop (ModuleDescription-only signals):
    auto mdStop = [&filter](edm::ModuleDescription const& md) { filter.stopOnThread(static_cast<int>(md.id())); };

    // Start lambda factories

    // ED stream-scoped signals (StreamContext + ModuleCallingContext)
    auto makeEdStreamStart = [&filter](std::string_view signal) {
      return [&filter, signal](edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
        if (not filter.tryCompactIndex(static_cast<int>(module_id(mcc)))) {
          return;
        }
        auto count = filter.event_.slots[sc.streamID().value()];
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, count);
      };
    };

    // ED global lumi-scoped signals (GlobalContext + ModuleCallingContext)
    auto makeEdGlobalLumiStart = [&filter](std::string_view signal) {
      return [&filter, signal](edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
        if (not filter.tryCompactIndex(static_cast<int>(module_id(mcc)))) {
          return;
        }
        auto count = filter.lumi_.slots[gc.luminosityBlockIndex().value()];
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, count);
      };
    };

    // ED global run-scoped signals (GlobalContext + ModuleCallingContext)
    auto makeEdGlobalRunStart = [&filter](std::string_view signal) {
      return [&filter, signal](edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
        if (not filter.tryCompactIndex(static_cast<int>(module_id(mcc)))) {
          return;
        }
        auto count = filter.run_.slots[gc.runIndex().value()];
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, count);
      };
    };

    // ED moduleBeginStream / moduleEndStream (stream ID is the counter)
    auto makeEdStreamOnceStart = [&filter](std::string_view signal) {
      return [&filter, signal](edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
        if (not filter.tryCompactIndex(static_cast<int>(module_id(mcc)))) {
          return;
        }
        auto count = static_cast<unsigned int>(sc.streamID().value());
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, count);
      };
    };

    // ED global once-per-module-per-job signals (GlobalContext + ModuleCallingContext, count=0)
    auto makeEdGlobalOnceStart = [&filter](std::string_view signal) {
      return [&filter, signal](auto const&, edm::ModuleCallingContext const& mcc) {
        if (not filter.tryCompactIndex(static_cast<int>(module_id(mcc)))) {
          return;
        }
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, 0u);
      };
    };

    // ED AccessInputProcessBlock (GlobalContext + ModuleCallingContext, per-module atomic counter)
    auto makeEdAccessInputProcessBlockStart = [&filter](std::string_view signal) {
      return [&filter, signal](auto const&, edm::ModuleCallingContext const& mcc) {
        auto idx = filter.tryCompactIndex(static_cast<int>(module_id(mcc)));
        if (not idx) {
          return;
        }
        auto count = filter.accessInputProcessBlockCounters_[*idx].fetch_add(1);
        filter.startOnThread(mcc.moduleDescription()->moduleLabel(), signal, count);
      };
    };

    // ED once-per-module signals (ModuleDescription only, count=0)
    auto makeEdOncePerModule = [&filter](std::string_view signal) {
      return [&filter, signal](edm::ModuleDescription const& md) {
        if (not filter.tryCompactIndex(static_cast<int>(md.id()))) {
          return;
        }
        filter.startOnThread(md.moduleLabel(), signal, 0u);
      };
    };

    // Source signals (count=0 — source profiling is not event/lumi/run-indexed)
    auto makeSourceStart = [&filter](std::string_view signal) {
      return [&filter, signal](auto const&) {
        if (not filter.tryCompactIndex(kSourceModuleID)) {
          return;
        }
        filter.startOnThread("source", signal, 0u);
      };
    };

    // --- ES Module construction ---
    //NOTE: we want the id to start at 1 not 0
    iAR.watchPreESModuleConstruction([&filter](auto const& desc) {
      if (not filter.tryCompactIndex(-1 * static_cast<int>(desc.id_ + 1))) {
        return;
      }
      std::string_view label = desc.label_.empty() ? desc.type_ : desc.label_;
      filter.startOnThread(label, "esModuleConstruction", 0u);
    });
    iAR.watchPostESModuleConstruction(
        [&filter](auto const& desc) { filter.stopOnThread(-1 * static_cast<int>(desc.id_ + 1)); });

    // --- ED Module construction / destruction ---
    iAR.watchPreModuleConstruction(makeEdOncePerModule("moduleConstruction"));
    iAR.watchPostModuleConstruction(mdStop);

    iAR.watchPreModuleDestruction(makeEdOncePerModule("moduleDestruction"));
    iAR.watchPostModuleDestruction(mdStop);

    // --- Source transitions ---
    iAR.watchPreSourceConstruction(makeSourceStart("sourceConstruction"));
    iAR.watchPostSourceConstruction([&filter](auto const&) { filter.stopOnThread(kSourceModuleID); });
    // using shared_ptr in order to have the lambda copyable as required by the ActivityRegistry
    iAR.watchPreOpenFile(
        [&filter, count = std::make_shared<std::atomic<unsigned int>>(0)](std::string const& fileName) {
          if (not filter.tryCompactIndex(kSourceModuleID)) {
            return;
          }
          filter.startOnThread("source", "openFile", count->fetch_add(1), fileName);
        });
    iAR.watchPostOpenFile([&filter](auto const&) { filter.stopOnThread(kSourceModuleID); });

    iAR.watchPreSourceEvent([&filter](edm::StreamID id) {
      // slot update is unconditional: module callbacks read it while this event runs
      filter.event_.slots[id.value()] = filter.event_.global.fetch_add(1);
      if (not filter.tryCompactIndex(kSourceModuleID)) {
        return;
      }
      filter.startOnThread("source", "sourceEvent", filter.event_.slots[id.value()]);
    });
    iAR.watchPostSourceEvent([&filter](edm::StreamID) { filter.stopOnThread(kSourceModuleID); });

    iAR.watchPreSourceRun([&filter](edm::RunIndex idx) {
      filter.run_.slots[idx.value()] = filter.run_.global.fetch_add(1);
      if (not filter.tryCompactIndex(kSourceModuleID)) {
        return;
      }
      filter.startOnThread("source", "sourceRun", filter.run_.slots[idx.value()]);
    });
    iAR.watchPostSourceRun([&filter](edm::RunIndex) { filter.stopOnThread(kSourceModuleID); });

    iAR.watchPreSourceLumi([&filter](edm::LuminosityBlockIndex idx) {
      filter.lumi_.slots[idx.value()] = filter.lumi_.global.fetch_add(1);
      if (not filter.tryCompactIndex(kSourceModuleID)) {
        return;
      }
      filter.startOnThread("source", "sourceLumi", filter.lumi_.slots[idx.value()]);
    });
    iAR.watchPostSourceLumi([&filter](edm::LuminosityBlockIndex) { filter.stopOnThread(kSourceModuleID); });

    iAR.watchPreSourceNextTransition([&filter, count = 0u]() mutable {
      if (not filter.tryCompactIndex(kSourceModuleID)) {
        return;
      }
      filter.startOnThread("source", "sourceNextTransition", count++);
    });
    iAR.watchPostSourceNextTransition([&filter]() { filter.stopOnThread(kSourceModuleID); });

    if (profileClearEvent) {
      iAR.watchPreClearEvent([&filter](edm::StreamContext const& sc) {
        auto count = filter.event_.slots[sc.streamID().value()];
        filter.startOnThread("ClearEvent", "clearEvent", count);
      });
      iAR.watchPostClearEvent([&filter](auto const&) { filter.stopOnThread(); });
    }

    // --- ED Module begin/end job ---
    iAR.watchPreModuleBeginJob(makeEdOncePerModule("moduleBeginJob"));
    iAR.watchPostModuleBeginJob(mdStop);
    iAR.watchPreModuleEndJob(makeEdOncePerModule("moduleEndJob"));
    iAR.watchPostModuleEndJob(mdStop);

    // --- ED Module stream transitions ---
    iAR.watchPreModuleBeginStream(makeEdStreamOnceStart("moduleBeginStream"));
    iAR.watchPostModuleBeginStream(edStop);
    iAR.watchPreModuleEndStream(makeEdStreamOnceStart("moduleEndStream"));
    iAR.watchPostModuleEndStream(edStop);
    iAR.watchPreModuleEvent(makeEdStreamStart("moduleEvent"));
    iAR.watchPostModuleEvent(edStop);
    iAR.watchPreModuleEventAcquire(makeEdStreamStart("moduleEventAcquire"));
    iAR.watchPostModuleEventAcquire(edStop);
    iAR.watchPreModuleEventDelayedGet(makeEdStreamStart("moduleEventDelayedGet"));
    iAR.watchPostModuleEventDelayedGet(edStop);
    iAR.watchPreEventReadFromSource(makeEdStreamStart("eventReadFromSource"));
    iAR.watchPostEventReadFromSource(edStop);
    iAR.watchPreModuleTransform(makeEdStreamStart("moduleTransform"));
    iAR.watchPostModuleTransform(edStop);
    iAR.watchPreModuleTransformAcquiring(makeEdStreamStart("moduleTransformAcquiring"));
    iAR.watchPostModuleTransformAcquiring(edStop);
    iAR.watchPreModuleStreamBeginRun(makeEdStreamStart("moduleStreamBeginRun"));
    iAR.watchPostModuleStreamBeginRun(edStop);
    iAR.watchPreModuleStreamEndRun(makeEdStreamStart("moduleStreamEndRun"));
    iAR.watchPostModuleStreamEndRun(edStop);
    iAR.watchPreModuleStreamBeginLumi(makeEdStreamStart("moduleStreamBeginLumi"));
    iAR.watchPostModuleStreamBeginLumi(edStop);
    iAR.watchPreModuleStreamEndLumi(makeEdStreamStart("moduleStreamEndLumi"));
    iAR.watchPostModuleStreamEndLumi(edStop);

    // --- ED Module global transitions ---
    iAR.watchPreModuleBeginProcessBlock(makeEdGlobalOnceStart("moduleBeginProcessBlock"));
    iAR.watchPostModuleBeginProcessBlock(edStop);
    iAR.watchPreModuleAccessInputProcessBlock(makeEdAccessInputProcessBlockStart("moduleAccessInputProcessBlock"));
    iAR.watchPostModuleAccessInputProcessBlock(edStop);
    iAR.watchPreModuleEndProcessBlock(makeEdGlobalOnceStart("moduleEndProcessBlock"));
    iAR.watchPostModuleEndProcessBlock(edStop);
    iAR.watchPreModuleGlobalBeginRun(makeEdGlobalRunStart("moduleGlobalBeginRun"));
    iAR.watchPostModuleGlobalBeginRun(edStop);
    iAR.watchPreModuleGlobalEndRun(makeEdGlobalRunStart("moduleGlobalEndRun"));
    iAR.watchPostModuleGlobalEndRun(edStop);
    iAR.watchPreModuleGlobalBeginLumi(makeEdGlobalLumiStart("moduleGlobalBeginLumi"));
    iAR.watchPostModuleGlobalBeginLumi(edStop);
    iAR.watchPreModuleGlobalEndLumi(makeEdGlobalLumiStart("moduleGlobalEndLumi"));
    iAR.watchPostModuleGlobalEndLumi(edStop);
    iAR.watchPreModuleWriteProcessBlock(makeEdGlobalOnceStart("moduleWriteProcessBlock"));
    iAR.watchPostModuleWriteProcessBlock(edStop);
    iAR.watchPreModuleWriteRun(makeEdGlobalRunStart("moduleWriteRun"));
    iAR.watchPostModuleWriteRun(edStop);
    iAR.watchPreModuleWriteLumi(makeEdGlobalLumiStart("moduleWriteLumi"));
    iAR.watchPostModuleWriteLumi(edStop);

    // ES Module transitions
    iAR.watchPreESModuleAcquire(
        [&filter](edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc) {
          auto idx = filter.tryCompactIndex(-1 * static_cast<int>(module_id(mcc) + 1));
          if (not idx) {
            return;
          }
          auto const& desc = *mcc.componentDescription();
          std::string_view label = desc.label_.empty() ? desc.type_ : desc.label_;
          auto count = filter.getAndStoreESAcquireCounter(*idx, mcc.callID());
          filter.startOnThread(label, "esModuleAcquire", count, iKey.name(), mcc.callID());
        });
    iAR.watchPostESModuleAcquire(esStop);
    iAR.watchPreESModule(
        [&filter](edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc) {
          auto idx = filter.tryCompactIndex(-1 * static_cast<int>(module_id(mcc) + 1));
          if (not idx) {
            return;
          }
          auto const& desc = *mcc.componentDescription();
          std::string_view label = desc.label_.empty() ? desc.type_ : desc.label_;
          auto count = filter.getESModuleCounter(*idx, mcc.callID());
          filter.startOnThread(label, "esModule", count, iKey.name(), mcc.callID());
        });
    iAR.watchPostESModule(esStop);
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
        profilerFilter_(moduleIDs_, moduleNames_.size(), filePattern_, config_) {
    if (moduleNames_.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "moduleNames must be non-empty: ModuleAllocProfiler is intended to profile individual modules, "
             "and profiling all modules at once would be too costly.";
    }

    if (not filePattern_.empty()) {
      if (not filePattern_.contains("%M"))
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%M'";
      if (not filePattern_.contains("%S"))
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%S'";
      if (not filePattern_.contains("%I"))
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%I'";
      if (not filePattern_.contains("%T"))
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%T'";
      if (not filePattern_.contains("%C"))
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%C'";
    }

    if (std::find(moduleNames_.begin(), moduleNames_.end(), "source") != moduleNames_.end()) {
      moduleIDs_.push_back(kSourceModuleID);
    }

    const bool profileClearEvent =
        std::find(moduleNames_.begin(), moduleNames_.end(), "@ClearEvent") != moduleNames_.end();

    iAR.watchPreModuleConstruction([this](edm::ModuleDescription const& description) {
      auto found = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleLabel());
      if (found != moduleNames_.end()) {
        moduleIDs_.push_back(static_cast<int>(description.id()));
        std::sort(moduleIDs_.begin(), moduleIDs_.end());
      }
    });

    iAR.watchPreESModuleConstruction([this](auto const& iDescription) {
      auto label = iDescription.label_;
      if (label.empty()) {
        label = iDescription.type_;
      }
      auto found = std::find(moduleNames_.begin(), moduleNames_.end(), label);
      if (found != moduleNames_.end()) {
        //NOTE: we want the id to start at 1 not 0
        int esID = -1 * static_cast<int>(iDescription.id_ + 1);
        moduleIDs_.push_back(esID);
        std::sort(moduleIDs_.begin(), moduleIDs_.end());
      }
    });

    // watchPreallocate must be registered before setupProfilerFile so allocateCounters()
    // is guaranteed to be called before any source signals fire.
    iAR.watchPreallocate([this](edm::service::SystemBounds const& b) {
      profilerFilter_.allocateCounters(
          b.maxNumberOfStreams(), b.maxNumberOfConcurrentLuminosityBlocks(), b.maxNumberOfConcurrentRuns());
    });

    if (nEventsToSkip_ > 0) {
      profilerFilter_.setGlobalKeep(false);
      iAR.watchPreSourceEvent([this](auto) {
        ++nEventsStarted_;
        if (nEventsStarted_ > nEventsToSkip_) {
          profilerFilter_.setGlobalKeep(true);
        }
      });
    }

    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();

    setupProfilerFile(iAR, profilerFilter_, profileClearEvent);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>())
        ->setComment(
            "List of ED/ES module labels to profile. Must be non-empty: the intent is to profile individual modules, "
            "and profiling all modules at once would be too costly. The ClearEvent signal is not tied to any module, "
            "but can be profiled including '@ClearEvent' in this list. The Source module can be profiled including "
            "'source' in this list.");
    ps.addUntracked<unsigned int>("nEventsToSkip", 0);
    ps.addUntracked<std::string>("filePattern", "")
        ->setComment(
            "Pattern for output file names. Must contain '%M' (module label), '%S' (signal name), "
            "'%I' (per-signal occurrence counter), '%T' (measurement type: 'alloc', 'atMaxActual', "
            "'added', 'dealloc', 'churn', 'churnalloc'), and '%C' (ESModule callID, 0 for non-ES signals). "
            "If empty (default), results are printed with MessageLogger.");
    ps.addUntracked<bool>("statistics", false)
        ->setComment("Whether to print some timing statistics about the memory measurement itself. Default is false.");
    ps.addUntracked<bool>("deallocationReport", true)
        ->setComment(
            "Whether to produce a report on deallocations. On deep stack traces this can take time, so turning it "
            "off could speed up if the deallocation report is not needed.");
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
