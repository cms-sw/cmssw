#ifndef PerfTools_AllocMonitor_plugins_AllocProfilerCommon_h
#define PerfTools_AllocMonitor_plugins_AllocProfilerCommon_h

#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MonitorStackNode.h"
#include "ThreadTracker.h"

#include <boost/algorithm/string.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <print>
#include <ranges>
#include <sstream>
#include <stacktrace>
#include <string>
#include <unordered_map>
#include <vector>

namespace cms::perftools::allocMon::profiler {

  struct AllocationRecord {
    void add(AllocationRecord const& o) {
      requested_ += o.requested_;
      actual_ += o.actual_;
      count_ += o.count_;
    }
    void subtract(AllocationRecord const& o) {
      requested_ -= o.requested_;
      actual_ -= o.actual_;
      count_ -= o.count_;
    }

    std::size_t requested_ = 0;
    std::size_t actual_ = 0;
    std::size_t count_ = 0;
  };

  struct DeallocationRecord {
    void add(DeallocationRecord const& o) {
      actual_ += o.actual_;
      count_ += o.count_;
    }
    void subtract(DeallocationRecord const& o) {
      actual_ -= o.actual_;
      count_ -= o.count_;
    }

    std::size_t actual_ = 0;
    std::size_t count_ = 0;
  };

  // Statistics Strategy Pattern
  class StatisticsStrategy {
  public:
    virtual ~StatisticsStrategy() = default;

    // RAII timer for measuring operation durations
    class ScopedTimer {
    public:
      ScopedTimer(std::chrono::duration<double, std::milli>* target) : target_(target) {
        if (target_) {
          start_ = std::chrono::steady_clock::now();
        }
      }
      ~ScopedTimer() {
        if (target_) {
          *target_ = std::chrono::steady_clock::now() - start_;
        }
      }

    private:
      std::chrono::duration<double, std::milli>* target_;
      std::chrono::steady_clock::time_point start_;
    };

    // Combined recording: timing and counting
    virtual ScopedTimer recordAllocation() = 0;
    virtual ScopedTimer recordDeallocation() = 0;

    // Allocation printing measurements
    virtual void recordAllocationsUniqueAggregated(std::size_t value) = 0;
    virtual ScopedTimer timeAllocationsAggregation() = 0;
    virtual ScopedTimer timeAllocationsSorting() = 0;
    virtual ScopedTimer timeAllocationsFormatting() = 0;
    virtual ScopedTimer timeAllocationsAtMaxActualSorting() = 0;
    virtual ScopedTimer timeAllocationsAtMaxActualFormatting() = 0;
    virtual ScopedTimer timeAllocationsAddedFiltering() = 0;
    virtual ScopedTimer timeAllocationsAddedFormatting() = 0;

    // Deallocation printing measurements
    virtual void recordDeallocationsUniqueAggregated(std::size_t value) = 0;
    virtual ScopedTimer timeDeallocationAggregation() = 0;
    virtual ScopedTimer timeDeallocationsSorting() = 0;
    virtual ScopedTimer timeDeallocationsFormatting() = 0;

    // Churn printing measurements
    virtual void recordChurnUniqueAggregated(std::size_t value) = 0;
    virtual void recordChurnMaxInnerSize(std::size_t value) = 0;
    virtual ScopedTimer timeChurnAggregation() = 0;
    virtual ScopedTimer timeChurnSorting() = 0;
    virtual ScopedTimer timeChurnFormatting() = 0;

    // Final statistics printout (accesses internal member data)
    virtual void printStatistics(edm::LogSystem& log,
                                 std::size_t numAllocTraces,
                                 std::size_t numDeallocTraces) const = 0;
  };

  // No-op strategy: zero overhead, no measurements
  class NoOpStatisticsStrategy : public StatisticsStrategy {
  public:
    ScopedTimer recordAllocation() override { return ScopedTimer(nullptr); }
    ScopedTimer recordDeallocation() override { return ScopedTimer(nullptr); }

    void recordAllocationsUniqueAggregated(std::size_t) override {}
    ScopedTimer timeAllocationsAggregation() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsSorting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsFormatting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsAtMaxActualSorting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsAtMaxActualFormatting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsAddedFiltering() override { return ScopedTimer(nullptr); }
    ScopedTimer timeAllocationsAddedFormatting() override { return ScopedTimer(nullptr); }

    void recordDeallocationsUniqueAggregated(std::size_t) override {}
    ScopedTimer timeDeallocationAggregation() override { return ScopedTimer(nullptr); }
    ScopedTimer timeDeallocationsSorting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeDeallocationsFormatting() override { return ScopedTimer(nullptr); }

    void recordChurnUniqueAggregated(std::size_t) override {}
    void recordChurnMaxInnerSize(std::size_t) override {}
    ScopedTimer timeChurnAggregation() override { return ScopedTimer(nullptr); }
    ScopedTimer timeChurnSorting() override { return ScopedTimer(nullptr); }
    ScopedTimer timeChurnFormatting() override { return ScopedTimer(nullptr); }

    void printStatistics(edm::LogSystem&, std::size_t, std::size_t) const override {}
  };

  // Collecting strategy: measures and reports all statistics
  class CollectingStatisticsStrategy : public StatisticsStrategy {
  public:
    ScopedTimer recordAllocation() override {
      ++numAllocations_;
      return ScopedTimer(&allocationsTime_);
    }
    ScopedTimer recordDeallocation() override {
      ++numDeallocations_;
      return ScopedTimer(&deallocationsTime_);
    }

    void recordAllocationsUniqueAggregated(std::size_t value) override { allocStats_.uniqueAggregated = value; }
    ScopedTimer timeAllocationsAggregation() override { return ScopedTimer(&allocStats_.t_aggregation); }
    ScopedTimer timeAllocationsSorting() override { return ScopedTimer(&allocStats_.t_sorting); }
    ScopedTimer timeAllocationsFormatting() override { return ScopedTimer(&allocStats_.t_format); }
    ScopedTimer timeAllocationsAtMaxActualSorting() override { return ScopedTimer(&allocStats_.t_atMaxActualSorting); }
    ScopedTimer timeAllocationsAtMaxActualFormatting() override {
      return ScopedTimer(&allocStats_.t_atMaxActualFormat);
    }
    ScopedTimer timeAllocationsAddedFiltering() override { return ScopedTimer(&allocStats_.t_addedFiltering); }
    ScopedTimer timeAllocationsAddedFormatting() override { return ScopedTimer(&allocStats_.t_addedFormat); }

    void recordDeallocationsUniqueAggregated(std::size_t value) override { deallocStats_.uniqueAggregated = value; }
    ScopedTimer timeDeallocationAggregation() override { return ScopedTimer(&deallocStats_.t_aggregation); }
    ScopedTimer timeDeallocationsSorting() override { return ScopedTimer(&deallocStats_.t_sorting); }
    ScopedTimer timeDeallocationsFormatting() override { return ScopedTimer(&deallocStats_.t_format); }

    void recordChurnUniqueAggregated(std::size_t value) override { churnStats_.uniqueAggregated = value; }
    void recordChurnMaxInnerSize(std::size_t value) override {
      churnStats_.maxInnerSize = std::max(churnStats_.maxInnerSize, value);
    }
    ScopedTimer timeChurnAggregation() override { return ScopedTimer(&churnStats_.t_aggregation); }
    ScopedTimer timeChurnSorting() override { return ScopedTimer(&churnStats_.t_sorting); }
    ScopedTimer timeChurnFormatting() override { return ScopedTimer(&churnStats_.t_format); }

    void printStatistics(edm::LogSystem& log, std::size_t numAllocTraces, std::size_t numDeallocTraces) const override;

  private:
    struct AllocPrintStats {
      std::chrono::duration<double, std::milli> t_aggregation{};
      std::chrono::duration<double, std::milli> t_sorting{};
      std::chrono::duration<double, std::milli> t_format{};
      std::chrono::duration<double, std::milli> t_atMaxActualSorting{};
      std::chrono::duration<double, std::milli> t_atMaxActualFormat{};
      std::chrono::duration<double, std::milli> t_addedFiltering{};
      std::chrono::duration<double, std::milli> t_addedFormat{};
      std::size_t uniqueAggregated = 0;
    };

    struct DeallocPrintStats {
      std::chrono::duration<double, std::milli> t_aggregation{};
      std::chrono::duration<double, std::milli> t_sorting{};
      std::chrono::duration<double, std::milli> t_format{};
      std::size_t uniqueAggregated = 0;
    };

    struct ChurnPrintStats {
      std::chrono::duration<double, std::milli> t_aggregation{};
      std::chrono::duration<double, std::milli> t_sorting{};
      std::chrono::duration<double, std::milli> t_format{};
      std::size_t uniqueAggregated = 0;
      std::size_t maxInnerSize = 0;
    };

    std::chrono::duration<double, std::milli> allocationsTime_{};
    std::chrono::duration<double, std::milli> deallocationsTime_{};
    std::size_t numAllocations_ = 0;
    std::size_t numDeallocations_ = 0;
    AllocPrintStats allocStats_;
    DeallocPrintStats deallocStats_;
    ChurnPrintStats churnStats_;
  };

  struct ReportConfiguration {
    bool printStatistics_ = false;
    bool deallocationReport_ = true;
    bool churnReport_ = true;
  };

  class StackNodeData;
  using MonitorStackNode = cms::perftools::allocMon::MonitorStackNode<StackNodeData>;

  class StackNodeData {
  public:
    StackNodeData(std::string filePattern, ReportConfiguration config);

    void recordAllocation(std::stacktrace trace, AllocationRecord record, void const* ptr);

    void recordDeallocation(std::stacktrace trace,
                            DeallocationRecord record,
                            void const* ptr,
                            MonitorStackNode* prevNode) {
      auto timer = statistics_->recordDeallocation();

      auto const deallocIndex = addDeallocation(std::move(trace), record);
      subtractDeallocationFromAllocations(deallocIndex, ptr, prevNode);
    }

    void recordDeallocationFromNestedMeasurement(void const* ptr, MonitorStackNode* prevNode) {
      subtractDeallocationFromAllocations({}, ptr, prevNode);
    }

    void print(edm::LogSystem& log, std::string_view measurementName) const;

  private:
    // If we get more than 2^32 different stack traces, we'll have other problems
    using UniqueTraceIndex = unsigned int;
    using TraceHashValue = std::size_t;

    struct AllocationTrace {
      AllocationTrace(AllocationRecord const& o, std::stacktrace trace)
          : records_{.total_ = o, .added_ = o, .atMaxActual_ = {}}, trace_(trace) {}

      void recordAllocation(AllocationRecord const& o) {
        records_.total_.add(o);
        records_.added_.add(o);
      }
      void recordDeallocation(AllocationRecord const& o) { records_.added_.subtract(o); }

      void setAtMaxActual() { records_.atMaxActual_ = records_.added_; }

      struct Records {
        void add(Records const& o) {
          total_.add(o.total_);
          added_.add(o.added_);
          atMaxActual_.add(o.atMaxActual_);
        }
        AllocationRecord total_;
        AllocationRecord added_;
        AllocationRecord atMaxActual_;
      };
      Records records_;

      std::stacktrace trace_;
    };

    struct DeallocationTrace {
      DeallocationTrace(DeallocationRecord const& o, std::stacktrace trace) : total_(o), trace_(trace) {}

      // Note: in DeallocationTrace we track the total number of deallocations for a given trace and hence call the
      // add() below. In other cases, e.g. in AllocationTrace, we track the net number of allocations (allocations minus
      // deallocations) and hence calls subtract() for deallocations.
      void recordDeallocation(DeallocationRecord const& o) { total_.add(o); }

      DeallocationRecord total_;
      std::stacktrace trace_;
    };

    struct ChurnRecord {
      UniqueTraceIndex deallocIndex_;
      AllocationRecord total_;
    };

    void printAllocations(edm::LogSystem& log,
                          std::string_view measurementName,
                          std::size_t commonTopEntries,
                          std::string_view commonTraceContext) const;

    void printDeallocations(edm::LogSystem& log,
                            std::string_view measurementName,
                            std::size_t commonTopEntries,
                            std::string_view commonTraceContext) const;

    void printChurn(edm::LogSystem& log,
                    std::string_view measurementName,
                    std::size_t commonTopEntries,
                    std::string_view commonTraceContext) const;

    void subtractDeallocationFromAllocations(std::optional<UniqueTraceIndex> deallocIndex,
                                             void const* ptr,
                                             MonitorStackNode* prevNode);

    void analyzeDeallocationForChurn(UniqueTraceIndex deallocIndex,
                                     AllocationRecord const& record,
                                     UniqueTraceIndex allocIndex);

    UniqueTraceIndex addDeallocation(std::stacktrace trace, DeallocationRecord record);

    std::string formatTrace(std::stacktrace const& trace, int skipFromBottom, int skipFromTop) const {
      assert(skipFromBottom < trace.size());
      assert(skipFromTop < trace.size());
      assert(skipFromBottom + skipFromTop < trace.size());
      return formatTrace(std::ranges::subrange(trace.cbegin() + skipFromBottom, trace.cend() - skipFromTop));
    }

    template <typename T>
    std::string formatTrace(T&& traceRange) const {
      std::string ret;
      for (auto const& [index, entry] : traceRange | std::views::enumerate) {
        ret += std::format("{:>4}# {} \n", index, entry);
      }
      return ret;
    }

    template <typename T, typename P>
    auto makeSortedIterators(T const& map, P&& pred) const {
      std::vector<typename T::const_iterator> ordered;
      ordered.reserve(map.size());
      for (auto it = map.begin(); it != map.end(); ++it) {
        ordered.push_back(it);
      }
      std::ranges::sort(ordered, std::greater{}, std::forward<P>(pred));
      return ordered;
    }

    // Compute the longest common set of topmost stack frames shared by all traces in the range.
    // Returns 0 if the range is empty or traces share no common frames.
    template <typename TraceRange1, typename TraceRange2>
    static std::size_t computeCommonTopEntries(TraceRange1&& first, TraceRange2&& second) {
      // Find an initial reference from whichever range is non-empty.
      auto it1 = std::ranges::begin(first);
      auto const end1 = std::ranges::end(first);
      auto it2 = std::ranges::begin(second);
      auto const end2 = std::ranges::end(second);

      if (it1 == end1 && it2 == end2) {
        return 0;
      }

      // Use the first available trace as reference.
      std::stacktrace const* refPtr = nullptr;
      if (it1 != end1) {
        refPtr = &(*it1);
        ++it1;
      } else {
        refPtr = &(*it2);
        ++it2;
      }
      std::stacktrace const& ref = *refPtr;
      std::size_t commonLen = ref.size();
      if (commonLen == 0) {
        return 0;
      }

      // If there is only one trace in total, there is no meaningful common context to strip.
      if (it1 == end1 && it2 == end2) {
        return 0;
      }

      // Cache for ref descriptions to avoid repeated calls to description() in the inner loop of intersectCommonTopEntries.
      std::vector<std::string> refDescriptionCache(ref.size());

      for (; it1 != end1 && commonLen > 0; ++it1) {
        commonLen = intersectCommonTopEntries(ref, refDescriptionCache, commonLen, *it1);
      }
      for (; it2 != end2 && commonLen > 0; ++it2) {
        commonLen = intersectCommonTopEntries(ref, refDescriptionCache, commonLen, *it2);
      }
      return commonLen;
    }

    // Helper: shrink commonLen to the length of the common suffix between ref and tr.
    static std::size_t intersectCommonTopEntries(std::stacktrace const& ref,
                                                 std::vector<std::string>& refDescriptionCache,
                                                 std::size_t commonLen,
                                                 std::stacktrace const& tr);

    template <typename F>
    void writeFileOrMessage(std::string_view measurementType,
                            std::string_view measurementName,
                            std::string_view commonTraceContext,
                            edm::LogSystem& log,
                            F&& format) const {
      if (not filePattern_.empty()) {
        auto fname = filePattern_;
        boost::replace_all(fname, "%T", measurementType);
        std::ofstream os(fname);
        std::print(os, "# {}\n", measurementName);
        if (not commonTraceContext.empty()) {
          std::print(os, "# Reported stack traces are based on\n");
          std::size_t pos = 0;
          while (pos < commonTraceContext.size()) {
            auto const nl = commonTraceContext.find('\n', pos);
            auto const line =
                commonTraceContext.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
            if (not line.empty()) {
              std::print(os, "# {}\n", line);
            }
            if (nl == std::string_view::npos) {
              break;
            }
            pos = nl + 1;
          }
        }
        format(os);
        os.close();
        log.format(" saved in {}", fname);
      } else {
        std::stringstream ss;
        format(ss);
        log << "\n" << ss.str();
      }
    }

    // About the measurement itself
    std::string filePattern_;
    std::unique_ptr<StatisticsStrategy> statistics_;
    ReportConfiguration config_;

    // Collection of stack traces
    long long presentActual_ = 0;
    long long maxActual_ = 0;
    std::vector<AllocationTrace> uniqueAllocTraces_;
    std::vector<DeallocationTrace> uniqueDeallocTraces_;
    std::unordered_map<TraceHashValue, UniqueTraceIndex> hashToAllocTraceIndex_;
    std::unordered_map<TraceHashValue, UniqueTraceIndex> hashToDeallocTraceIndex_;
    std::unordered_map<void const*, std::tuple<AllocationRecord, UniqueTraceIndex>> addressToAllocation_;

    // Same indexing as in uniqueAllocTraces_
    // Assuming the number of different deallocation traces per one
    // allocating trace would be small enough that a linear search is
    // good enough
    std::vector<std::vector<ChurnRecord>> churnRecords_;
  };

  inline std::unique_ptr<MonitorStackNode>& currentMonitorStackNode() {
    static thread_local std::unique_ptr<MonitorStackNode> ptr;
    return ptr;
  }

  inline bool& threadActiveMonitoring() {
    static bool s_active[cms::perftools::allocMon::ThreadTracker::kTotalEntries]{};
    return s_active[cms::perftools::allocMon::ThreadTracker::instance().thread_index()];
  }

}  // namespace cms::perftools::allocMon::profiler

#endif  // C++ version checks
#endif  // PerfTools_AllocMonitor_plugins_AllocProfilerData_h
