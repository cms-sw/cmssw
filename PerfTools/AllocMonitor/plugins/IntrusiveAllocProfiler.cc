#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include "MonitorStackNode.h"
#include "ThreadTracker.h"

#include <boost/algorithm/string.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <print>
#include <ranges>
#include <stacktrace>
#include <unordered_map>

namespace {
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

    void printStatistics(edm::LogSystem& log, std::size_t numAllocTraces, std::size_t numDeallocTraces) const override {
      log.format("\nStatistics:\nAllocations recorded {} in {}\nDeallocations recorded {} in {}\n",
                 numAllocations_,
                 allocationsTime_,
                 numDeallocations_,
                 deallocationsTime_);
      log.format("Allocation traces total {} aggregated {}; aggregation {} sorting {} formatting {}\n",
                 numAllocTraces,
                 allocStats_.uniqueAggregated,
                 allocStats_.t_aggregation,
                 allocStats_.t_sorting,
                 allocStats_.t_format);
      log.format(
          "AtMaxActual sorting {} formatting {}\n", allocStats_.t_atMaxActualSorting, allocStats_.t_atMaxActualFormat);
      log.format("Added filtering {} formatting {}\n", allocStats_.t_addedFiltering, allocStats_.t_addedFormat);
      log.format("Deallocation traces total {} aggregated {}; aggregation {} sorting {} formatting {}\n",
                 numDeallocTraces,
                 deallocStats_.uniqueAggregated,
                 deallocStats_.t_aggregation,
                 deallocStats_.t_sorting,
                 deallocStats_.t_format);
      log.format("Churn aggregated traces {} inner container max size {}; aggregation {} sorting {} formatting {}",
                 churnStats_.uniqueAggregated,
                 churnStats_.maxInnerSize,
                 churnStats_.t_aggregation,
                 churnStats_.t_sorting,
                 churnStats_.t_format);
    }

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

  class StackNodeData;
  using MonitorStackNode = cms::perftools::allocMon::MonitorStackNode<StackNodeData>;

  class StackNodeData {
  public:
    StackNodeData(std::stacktrace trace, unsigned int fileCount, std::string filePattern, bool printStatistics)
        : trace_(std::move(trace)),
          filePattern_(std::move(filePattern)),
          statistics_(printStatistics ? static_cast<std::unique_ptr<StatisticsStrategy>>(
                                            std::make_unique<CollectingStatisticsStrategy>())
                                      : std::make_unique<NoOpStatisticsStrategy>()) {
      // Skip first entries that are internals for AllocMonitor if they are in the trace
      // By experience these entries are not always part of the stack trace
      auto it = trace_.cbegin();
      // The startOnThread() is a member of MonitorAdaptor, but with
      // debug synbols the MonitorAdaptor does not seem to be part of
      // the symbol name
      if (it->description().contains("startOnThread")) {
        ++it;
        ++startFromEntry_;
      }
      if (it->description().contains("IntrusiveAllocProfiler::start")) {
        ++it;
        ++startFromEntry_;
      }
      if (it->description().contains("edm::IntrusiveMonitorBase::startMonitoring")) {
        ++it;
        ++startFromEntry_;
      }
      assert(it != trace.cend());
      stackDepth_ = std::distance(it, trace_.cend());

      if (not filePattern_.empty()) {
        boost::replace_all(filePattern_, "%I", std::to_string(fileCount));
      }
    }

    std::size_t stackDepth() const { return stackDepth_; }

    std::stacktrace const& stacktrace() const { return trace_; }

    void recordAllocation(std::stacktrace trace, AllocationRecord record, void const* ptr) {
      auto timer = statistics_->recordAllocation();

      UniqueTraceIndex traceIndex;
      auto hash = std::hash<std::stacktrace>()(trace);
      auto found = hashToAllocTraceIndex_.find(hash);
      if (found != hashToAllocTraceIndex_.end()) {
        traceIndex = found->second;
        assert(trace == uniqueAllocTraces_[traceIndex].trace_);
        uniqueAllocTraces_[traceIndex].recordAllocation(record);
      } else {
        // intentionally narrowing
        traceIndex = static_cast<UniqueTraceIndex>(uniqueAllocTraces_.size());
        uniqueAllocTraces_.emplace_back(record, std::move(trace));
        hashToAllocTraceIndex_.emplace(hash, traceIndex);
      }
      addressToAllocation_.emplace(ptr, std::tuple{record, traceIndex});

      presentActual_ += record.actual_;
      if (presentActual_ > maxActual_) {
        for (auto& trace : uniqueAllocTraces_) {
          trace.setAtMaxActual();
        }
        maxActual_ = presentActual_;
      }
    }

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

    template <typename T>
    void print(T&& log, std::string_view measurementName) const {
      log.format(" Reported stack traces are based on\n{}", formatTrace(trace_, startFromEntry_, 0));

      printAllocations(log, measurementName);
      printDeallocations(log, measurementName);
      printChurn(log, measurementName);

      statistics_->printStatistics(log, uniqueAllocTraces_.size(), uniqueDeallocTraces_.size());
    }

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

      void recordDeallocation(DeallocationRecord const& o) { total_.add(o); }

      DeallocationRecord total_;
      std::stacktrace trace_;
    };

    struct ChurnRecord {
      UniqueTraceIndex deallocIndex_;
      AllocationRecord total_;
    };

    template <typename T>
    void printAllocations(T&& log, std::string_view measurementName) const {
      using AggregationMap = std::unordered_map<std::string, AllocationTrace::Records>;

      AggregationMap aggregatedAllocs;
      {
        auto timer = statistics_->timeAllocationsAggregation();
        for (auto const& record : uniqueAllocTraces_) {
          aggregatedAllocs[formatTrace(record.trace_, 0, stackDepth_ - 1)].add(record.records_);
        }
        statistics_->recordAllocationsUniqueAggregated(aggregatedAllocs.size());
      }

      std::vector<std::unordered_map<std::string, AllocationTrace::Records>::const_iterator> orderAllocs;
      {
        auto timer = statistics_->timeAllocationsSorting();
        orderAllocs = makeSortedIterators(aggregatedAllocs, [](auto const& it) { return it->second.total_.actual_; });
      }

      {
        auto timer = statistics_->timeAllocationsFormatting();
        log.format("\nAll allocations");
        writeFileOrMessage("alloc", measurementName, log, [&](std::ostream& os) {
          for (auto const& itTrace : orderAllocs) {
            auto const& records = itTrace->second;
            std::print(os,
                       "count {} requested {} actual {}\n{}",
                       records.total_.count_,
                       records.total_.requested_,
                       records.total_.actual_,
                       itTrace->first);
          }
        });
      }

      // AtMaxActual
      {
        auto timer = statistics_->timeAllocationsAtMaxActualSorting();
        std::ranges::sort(orderAllocs, std::greater{}, [](auto const& it) { return it->second.atMaxActual_.actual_; });
      }

      {
        auto timer = statistics_->timeAllocationsAtMaxActualFormatting();
        log.format("\nAllocated memory at the max actual moment");
        writeFileOrMessage("atMaxActual", measurementName, log, [&](std::ostream& os) {
          for (auto const& itTrace : orderAllocs) {
            auto const& records = itTrace->second;
            std::print(os,
                       "count {} requested {} actual {}\n{}",
                       records.atMaxActual_.count_,
                       records.atMaxActual_.requested_,
                       records.atMaxActual_.actual_,
                       itTrace->first);
          }
        });
      }

      // Added
      {
        auto timer = statistics_->timeAllocationsAddedFiltering();
        std::erase_if(orderAllocs, [](auto const& it) { return it->second.added_.count_ == 0; });
        std::ranges::sort(orderAllocs, std::greater{}, [](auto const& it) { return it->second.added_.actual_; });
      }

      {
        auto timer = statistics_->timeAllocationsAddedFormatting();
        log.format("\nAdded memory");
        writeFileOrMessage("added", measurementName, log, [&](std::ostream& os) {
          for (auto const& itTrace : orderAllocs) {
            auto const& records = itTrace->second;
            std::print(os,
                       "count {} requested {} actual {}\n{}",
                       records.added_.count_,
                       records.added_.requested_,
                       records.added_.actual_,
                       itTrace->first);
            // The keys of aggregatedAllocs are large, so destruct them as soon as possible. The erase() does not invalidate
            // the iterators to other elements, so it's safe to erase while iterating through orderAllocs.
            aggregatedAllocs.erase(itTrace);
          }
        });
      }
    }

    template <typename T>
    void printDeallocations(T&& log, std::string_view measurementName) const {
      using AggregationMap = std::unordered_map<std::string, DeallocationRecord>;
      AggregationMap aggregatedDeallocs;

      {
        auto timer = statistics_->timeDeallocationAggregation();
        for (auto const& record : uniqueDeallocTraces_) {
          aggregatedDeallocs[formatTrace(record.trace_, 0, stackDepth_ - 1)].add(record.total_);
        }
        statistics_->recordDeallocationsUniqueAggregated(aggregatedDeallocs.size());
      }

      std::vector<std::unordered_map<std::string, DeallocationRecord>::const_iterator> orderDeallocs;
      {
        auto timer = statistics_->timeDeallocationsSorting();
        orderDeallocs = makeSortedIterators(aggregatedDeallocs, [](auto const& it) { return it->second.actual_; });
      }

      {
        auto timer = statistics_->timeDeallocationsFormatting();
        log.format("\nAll deallocations");
        writeFileOrMessage("dealloc", measurementName, log, [&](std::ostream& os) {
          for (auto const& itTrace : orderDeallocs) {
            auto const& record = itTrace->second;
            std::print(os, "count {} actual {}\n{}", record.count_, record.actual_, itTrace->first);
            aggregatedDeallocs.erase(itTrace);
          }
        });
      }
    }

    template <typename T>
    void printChurn(T&& log, std::string_view measurementName) const {
      using AggregationMap = std::unordered_map<std::string, AllocationRecord>;
      AggregationMap aggregatedChurn;
      AggregationMap aggregatedChurnAllocs;

      {
        auto timer = statistics_->timeChurnAggregation();
        auto traceToString = [](auto const& traceEntry) { return traceEntry.description(); };
        // zip stops at the shorter range, naturally handling churnRecords_ being possibly smaller than uniqueAllocTraces_
        for (auto const& [allocRecord, churnVec] : std::views::zip(uniqueAllocTraces_, churnRecords_)) {
          auto const& allocTrace = allocRecord.trace_;
          // can skip the topmost stackDepth entries because they are the same throughout the measurement
          // subtract 1 in case the measurement starting function is doing the churn
          auto const allocTraceStrings = allocTrace | std::views::take(allocTrace.size() - (stackDepth_ - 1)) |
                                         std::views::transform(traceToString) |
                                         std::ranges::to<std::vector<std::string>>();

          statistics_->recordChurnMaxInnerSize(churnVec.size());
          for (auto const& record : churnVec) {
            // Stopping to the lowest common stacktrace_entry between allocation and deallocation
            {
              auto const& deallocTrace = uniqueDeallocTraces_[record.deallocIndex_].trace_;
              auto const deallocTraceStrings =
                  deallocTrace | std::views::take(deallocTrace.size() - (stackDepth_ - 1)) |
                  std::views::transform(traceToString) | std::ranges::to<std::vector<std::string>>();

              // iterate from the top (outermost) downward to find where the traces first diverge
              auto allocEntries = allocTraceStrings | std::views::reverse;
              auto deallocEntries = deallocTraceStrings | std::views::reverse;

              auto [it_alloc, it_dealloc] = std::ranges::mismatch(allocEntries, deallocEntries);
              assert(it_alloc != allocEntries.end() and it_dealloc != deallocEntries.end());

              // Because the comparisons were done with strings, the
              // it_alloc and it_dealloc should point to different
              // functions. Go one level up (toward outer frames) to find the common function
              assert(it_alloc != allocEntries.begin());
              --it_alloc;

              auto str_trace = formatTrace(std::ranges::subrange(std::prev(it_alloc.base()), allocTraceStrings.end()));
              aggregatedChurn[str_trace].add(record.total_);
            }
            // Churn allocation stack traces
            {
              auto str_trace = formatTrace(allocTraceStrings);
              aggregatedChurnAllocs[str_trace].add(record.total_);
            }
          }
        }
        statistics_->recordChurnUniqueAggregated(aggregatedChurn.size());
      }

      std::vector<std::unordered_map<std::string, AllocationRecord>::const_iterator> orderChurn, orderChurnAllocs;
      {
        auto timer = statistics_->timeChurnSorting();
        orderChurn = makeSortedIterators(aggregatedChurn, [](auto const& it) { return it->second.count_; });
        orderChurnAllocs = makeSortedIterators(aggregatedChurnAllocs, [](auto const& it) { return it->second.count_; });
      }

      {
        auto timer = statistics_->timeChurnFormatting();
        log.format("\nMemory allocation+deallocation churn");
        writeFileOrMessage("churn", measurementName, log, [&](std::ostream& os) {
          for (auto const& it : orderChurn) {
            auto const& record = it->second;
            std::print(
                os, "count {} requested {} actual {}\n{}", record.count_, record.requested_, record.actual_, it->first);
            aggregatedChurn.erase(it);
          }
        });
        log.format("\nMemory allocation+deallocation churn allocations");
        writeFileOrMessage("churnalloc", measurementName, log, [&](std::ostream& os) {
          for (auto const& it : orderChurnAllocs) {
            auto const& record = it->second;
            std::print(
                os, "count {} requested {} actual {}\n{}", record.count_, record.requested_, record.actual_, it->first);
            aggregatedChurnAllocs.erase(it);
          }
        });
      }
    }

    void subtractDeallocationFromAllocations(std::optional<UniqueTraceIndex> deallocIndex,
                                             void const* ptr,
                                             MonitorStackNode* prevNode) {
      auto found = addressToAllocation_.find(ptr);
      if (found == addressToAllocation_.end()) {
        // The memory was allocated before the measurement region
        // If there is an earlier node in the stack, try to subtract it from there
        if (prevNode) {
          prevNode->get().recordDeallocationFromNestedMeasurement(ptr, prevNode->previousNode());
        }
        // If there is no earlier node in the stack, ignore the deallocation
        return;
      }
      // subtraction
      auto const& allocRecord = std::get<AllocationRecord>(found->second);
      auto const allocIndex = std::get<UniqueTraceIndex>(found->second);
      uniqueAllocTraces_[allocIndex].recordDeallocation(allocRecord);

      // Account the presentActual on the node where the memory was allocated
      // This can lead to some possibly weird cases when a nested node deallocates memory
      presentActual_ -= allocRecord.actual_;

      // churn analysis (only if deallocation is from the same measurement and not from a nested one)
      if (deallocIndex) {
        analyzeDeallocationForChurn(*deallocIndex, allocRecord, allocIndex);
      }

      addressToAllocation_.erase(found);
    }

    void analyzeDeallocationForChurn(UniqueTraceIndex deallocIndex,
                                     AllocationRecord const& record,
                                     UniqueTraceIndex allocIndex) {
      // allocIndex points to an element in uniqueAllocTraces_ and is thus at most the size of uniqueAllocTraces_
      if (allocIndex >= churnRecords_.size()) {
        churnRecords_.resize(uniqueAllocTraces_.size());
      }
      auto& churns = churnRecords_[allocIndex];
      auto found =
          std::ranges::find_if(churns, [deallocIndex](auto& elem) { return elem.deallocIndex_ == deallocIndex; });
      if (found == churns.end()) {
        churns.push_back(ChurnRecord{.deallocIndex_ = deallocIndex, .total_ = record});
      } else {
        found->total_.add(record);
      }
    }

    UniqueTraceIndex addDeallocation(std::stacktrace trace, DeallocationRecord record) {
      auto hash = std::hash<std::stacktrace>()(trace);
      auto found = hashToDeallocTraceIndex_.find(hash);
      UniqueTraceIndex traceIndex;
      if (found != hashToDeallocTraceIndex_.end()) {
        traceIndex = found->second;
        assert(trace == uniqueDeallocTraces_[traceIndex].trace_);
        uniqueDeallocTraces_[traceIndex].recordDeallocation(record);
      } else {
        traceIndex = uniqueDeallocTraces_.size();
        uniqueDeallocTraces_.emplace_back(record, std::move(trace));
        hashToDeallocTraceIndex_.emplace(hash, traceIndex);
      }
      return traceIndex;
    }

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

    template <typename T, typename F>
    void writeFileOrMessage(std::string_view measurementType,
                            std::string_view measurementName,
                            T&& log,
                            F&& format) const {
      if (not filePattern_.empty()) {
        auto fname = filePattern_;
        boost::replace_all(fname, "%M", measurementType);
        std::ofstream os(fname);
        std::print(os, "# {}\n", measurementName);
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
    std::stacktrace trace_;
    int startFromEntry_ = 0;
    std::size_t stackDepth_ = 0;
    std::string filePattern_;
    std::unique_ptr<StatisticsStrategy> statistics_;

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
  std::unique_ptr<MonitorStackNode>& currentMonitorStackNode() {
    static thread_local std::unique_ptr<MonitorStackNode> ptr;
    return ptr;
  }

  std::atomic<unsigned int>& globalFileCounter() {
    static std::atomic<unsigned int> counter = 0;
    return counter;
  }

  bool& threadActiveMonitoring() {
    using namespace cms::perftools::allocMon;
    static bool s_active[ThreadTracker::kTotalEntries]{};
    return s_active[ThreadTracker::instance().thread_index()];
  }

  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread(std::string_view name, std::string filePattern, bool printStatistics) {
      threadActiveMonitoring() = false;
      auto trace = std::stacktrace::current();
      auto const fileCount = globalFileCounter().fetch_add(1);
      auto node = std::make_unique<MonitorStackNode>(
          name,
          std::move(currentMonitorStackNode()),
          StackNodeData(std::move(trace), fileCount, std::move(filePattern), printStatistics));
      {
        edm::LogSystem log("IntrusiveAllocProfiler");
        using namespace cms::perftools::allocMon;
        log.format("Starting tracing for \"{}\".", name);
      }
      currentMonitorStackNode() = std::move(node);
      threadActiveMonitoring() = true;
    }
    static void stopOnThread(std::string_view name) {
      threadActiveMonitoring() = false;
      using namespace cms::perftools::allocMon;
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
        printStatistics_(iPSet.getUntrackedParameter<bool>("statistics")) {
    if (not pattern_.empty()) {
      if (not pattern_.contains("%I")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%I'";
      }
      if (not pattern_.contains("%M")) {
        throw edm::Exception(edm::errors::Configuration) << "filePattern did not contain '%M'";
      }
    }

    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();
  };
  ~IntrusiveAllocProfiler() noexcept override = default;

  void start(std::string_view name, bool nameIsString) final {
    MonitorAdaptor::startOnThread(name, pattern_, printStatistics_);
  }
  void stop(std::string_view name) noexcept final { MonitorAdaptor::stopOnThread(name); }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::string>("filePattern", "")
        ->setComment(
            "Pattern for the file names for the measurement results. Must contain '%I' for the counter of different "
            "files, and '%M' for the measurement type (that are 'alloc', 'dealloc', 'atMaxAlloc', 'added', 'churn', "
            "'churnalloc'). If empty (default), results are printed with MessageLogger.");
    ps.addUntracked<bool>("statistics", false)
        ->setComment("Whether to print some timing statistics about the memory measurement itself. Default is false.");
    iDesc.addDefault(ps);
  }

private:
  std::string pattern_;
  bool printStatistics_ = false;
};

typedef edm::serviceregistry::ParameterSetMaker<edm::IntrusiveMonitorBase, IntrusiveAllocProfiler>
    IntrusiveAllocProfilerMaker;
DEFINE_FWK_SERVICE_MAKER(IntrusiveAllocProfiler, IntrusiveAllocProfilerMaker);

#endif  // C++ version checks
